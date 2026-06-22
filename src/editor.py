"""
editor.py
Main pipeline orchestrator.
Chains: audio_extractor → transcriber → silence_remover → hook_detector
        → subtitle_renderer → music_mixer → clip_exporter
"""
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class EditConfig:
    # Input / Output
    input_path: str = ""
    output_path: str = ""
    output_dir: str = "output"

    # Clip mode
    mode: Literal["short", "long"] = "long"
    duration: float = 60.0
    size: str = "16:9"
    fps: int = 30

    # Noise reduction
    denoise: bool = False
    denoise_postfilter: bool = False
    deep_filter_bin: str | None = None
    denoise_workers: int | None = None

    # Silence removal
    remove_silence: bool = False
    silence_thresh_db: float = -40.0
    min_silence_ms: int = 500

    # Hook detection
    find_hook: bool = False
    hook_duration: float = 30.0
    hook_ai: bool = True

    # Subtitles
    add_subtitle: bool = False
    subtitle_style: str = "default"
    subtitle_language: str | None = None
    whisper_model: str = "base"
    srt_path: str | None = None

    # Background music
    add_music: bool = False
    music_path: str | None = None
    music_volume: float = 0.15
    music_ducking: bool = True

    # Quality
    crf: int = 23
    audio_bitrate: str = "192k"


@dataclass
class EditResult:
    success: bool = False
    output_path: str = ""
    audio_path: str = ""
    transcript_segments: list = field(default_factory=list)
    hook: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    steps_completed: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def run_pipeline(cfg: EditConfig) -> EditResult:
    """
    Execute the full editing pipeline based on the config.

    Pipeline order:
        0. Noise reduction   (if denoise — DeepFilterNet)
        1. Audio extraction  (always)
        2. Transcription     (if hook+ai — original audio for hook scoring)
        3. Silence removal   (if remove_silence)
        4. Hook detection    (if find_hook)
        5. Subtitle rendering(if add_subtitle — transcribe working_video for correct timeline)
        6. Music mixing      (if add_music)
        7. Clip export       (always — resize + final output)

    Returns:
        EditResult with paths, transcript, hook info, timing.
    """
    from src.audio_extractor import extract_audio, check_ffmpeg
    from src.transcriber import transcribe, load_segments_from_json
    from src.silence_remover import remove_silence
    from src.hook_detector import detect_hook, extract_hook_clip
    from src.subtitle_renderer import render_subtitles, burn_srt_file
    from src.music_mixer import mix_music
    from src.clip_exporter import export_clip, get_video_info

    t_start = time.time()
    result = EditResult()

    if not check_ffmpeg():
        result.errors.append("FFmpeg not found on PATH. Please install FFmpeg and add it to PATH.")
        return result

    input_path = Path(cfg.input_path).resolve()
    if not input_path.exists():
        result.errors.append(f"Input file not found: {input_path}")
        return result

    if cfg.output_path:
        output_path = Path(cfg.output_path).resolve()
    else:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        suffix = "_short" if cfg.mode == "short" else "_edited"
        output_path = out_dir / f"{stem}{suffix}.mp4"

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    working_video = str(input_path)
    segments: list[dict] = []
    sub_segments: list[dict] = []

    # ── Step 0: Noise reduction ─────────────────────────────────────────────────
    if cfg.denoise:
        print("\n" + "=" * 60)
        print("Step 0 — Noise Reduction (DeepFilterNet)")
        print("=" * 60)
        from src.noise_reducer import denoise_video

        denoised_out = str(temp_dir / f"{input_path.stem}_denoised.mp4")
        working_video = denoise_video(
            working_video,
            denoised_out,
            temp_dir=temp_dir,
            postfilter=cfg.denoise_postfilter,
            deep_filter_bin=cfg.deep_filter_bin,
            workers=cfg.denoise_workers,
        )
        result.steps_completed.append("noise_reduction")
    else:
        print("\nStep 0 — Noise Reduction: SKIPPED")

    # ── Step 1: Extract audio ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1/7 — Audio Extraction")
    print("=" * 60)
    audio_wav = temp_dir / f"{Path(working_video).stem}.wav"
    audio_path = extract_audio(
        working_video,
        str(audio_wav),
        force=cfg.denoise,
    )
    result.audio_path = audio_path
    result.steps_completed.append("audio_extraction")

    # ── Step 2: Transcribe (hook AI only — original audio) ─────────────────────
    if cfg.find_hook and cfg.hook_ai:
        print("\n" + "=" * 60)
        print("Step 2/7 — Transcription (Whisper, for hook AI)")
        print("=" * 60)
        tr = transcribe(
            audio_path,
            model_name=cfg.whisper_model,
            language=cfg.subtitle_language,
            output_dir=str(temp_dir),
            formats=["srt", "json"],
        )
        segments = tr["segments"]
        result.transcript_segments = segments
        result.steps_completed.append("transcription")
    else:
        print("\nStep 2/7 — Transcription: SKIPPED")

    # ── Step 3: Silence removal ─────────────────────────────────────────────────
    if cfg.remove_silence:
        print("\n" + "=" * 60)
        print("Step 3/7 — Silence Removal")
        print("=" * 60)
        silence_out = str(temp_dir / f"{input_path.stem}_nosilence.mp4")
        working_video = remove_silence(
            working_video,
            audio_path,
            silence_out,
            silence_thresh_db=cfg.silence_thresh_db,
            min_silence_ms=cfg.min_silence_ms,
            crf=cfg.crf,
        )
        result.steps_completed.append("silence_removal")
    else:
        print("\nStep 3/7 — Silence Removal: SKIPPED")

    # ── Step 4: Hook detection ──────────────────────────────────────────────────
    if cfg.find_hook:
        print("\n" + "=" * 60)
        print("Step 4/7 — Hook Detection")
        print("=" * 60)
        hook_result = detect_hook(
            audio_path,
            segments=segments if segments else None,
            target_duration=cfg.hook_duration,
            use_ai=cfg.hook_ai,
        )
        result.hook = hook_result
        hook_out = str(temp_dir / f"{input_path.stem}_hook.mp4")
        working_video = extract_hook_clip(working_video, hook_result, hook_out)
        result.steps_completed.append("hook_detection")
    else:
        print("\nStep 4/7 — Hook Detection: SKIPPED")

    # ── Step 5: Subtitle rendering ──────────────────────────────────────────────
    if cfg.add_subtitle:
        print("\n" + "=" * 60)
        print("Step 5/7 — Subtitle Rendering")
        print("=" * 60)
        sub_out = str(temp_dir / f"{input_path.stem}_subbed.mp4")

        if cfg.srt_path and Path(cfg.srt_path).exists():
            print(f"[editor] Using provided SRT: {cfg.srt_path}")
            from src.subtitle_renderer import _parse_srt as srt_parse
            sub_segments = srt_parse(cfg.srt_path)
        else:
            sub_audio = extract_audio(
                working_video,
                str(temp_dir / f"{input_path.stem}_subaudio.wav"),
                force=True,
            )
            tr = transcribe(
                sub_audio,
                model_name=cfg.whisper_model,
                language=cfg.subtitle_language,
                output_dir=str(temp_dir),
                formats=["srt", "json"],
                vad_filter=False,
                beam_size=5,
            )
            sub_segments = tr["segments"]
            cfg.srt_path = tr.get("srt_path")
            result.transcript_segments = sub_segments
            print(f"[editor] Transcribed {len(sub_segments)} subtitle segments")

        if sub_segments:
            sub_target = str(output_path) if not cfg.add_music and cfg.mode == "long" else sub_out
            working_video = render_subtitles(
                working_video,
                sub_segments,
                sub_target,
                style=cfg.subtitle_style,
                language=cfg.subtitle_language,
                crf=cfg.crf,
            )
            if sub_target == str(output_path):
                result.steps_completed.append("subtitle_rendering")
                result.steps_completed.append("export")
                result.output_path = str(output_path)
                result.elapsed_sec = round(time.time() - t_start, 1)
                result.success = True
                print("\n" + "=" * 60)
                print(f"DONE in {result.elapsed_sec}s")
                print(f"Output: {result.output_path}")
                print(f"Steps: {' -> '.join(result.steps_completed)}")
                print("=" * 60)
                return result
            result.steps_completed.append("subtitle_rendering")
        else:
            print("[editor] No subtitle segments found — skipping burn-in.")
    else:
        print("\nStep 5/7 — Subtitle Rendering: SKIPPED")

    # ── Step 6: Music mixing ────────────────────────────────────────────────────
    if cfg.add_music and cfg.music_path:
        print("\n" + "=" * 60)
        print("Step 6/7 — Music Mixing")
        print("=" * 60)
        music_out = str(temp_dir / f"{input_path.stem}_music.mp4")
        working_video = mix_music(
            working_video,
            cfg.music_path,
            music_out,
            music_volume=cfg.music_volume,
            ducking=cfg.music_ducking,
            speech_segments=sub_segments if cfg.music_ducking and sub_segments else segments if cfg.music_ducking else None,
        )
        result.steps_completed.append("music_mixing")
    else:
        print("\nStep 6/7 — Music Mixing: SKIPPED")

    # ── Step 7: Export ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7/7 — Clip Export")
    print("=" * 60)

    start_time = 0.0
    if cfg.mode == "short" and not cfg.find_hook:
        start_time = 0.0

    final_path = export_clip(
        working_video,
        str(output_path),
        mode=cfg.mode if not cfg.find_hook else "long",
        size=cfg.size,
        duration=cfg.duration if cfg.mode == "short" else None,
        start_time=start_time,
        fps=cfg.fps,
        crf=cfg.crf,
        audio_bitrate=cfg.audio_bitrate,
    )
    result.steps_completed.append("export")
    result.output_path = final_path
    result.elapsed_sec = round(time.time() - t_start, 1)
    result.success = True

    print("\n" + "=" * 60)
    print(f"DONE in {result.elapsed_sec}s")
    print(f"Output: {result.output_path}")
    print(f"Steps: {' -> '.join(result.steps_completed)}")
    print("=" * 60)

    return result
