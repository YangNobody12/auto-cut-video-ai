"""
main.py — Auto Cut AI
CLI entry point for the video editing pipeline.

Usage examples:
  # Full edit with silence removal + subtitles:
  python main.py --input video.mp4 --remove-silence --subtitle

  # Short hook clip (30s), TikTok format, with music:
  python main.py --input video.mp4 --mode short --duration 30 --hook --size 9:16 --music bgm.mp3

  # Long clip, subtitle style karaoke, add background music:
  python main.py --input video.mp4 --subtitle --sub-style karaoke --music bgm.mp3 --music-volume 0.2

  # Use AI to find hook (requires .env key):
  python main.py --input video.mp4 --hook --hook-ai --mode short --duration 60

  # Transcribe only:
  python main.py --input video.mp4 --transcribe-only --whisper-model medium
"""
import argparse
import sys
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auto-cut-ai",
        description="Auto Cut AI — Python video editor with silence removal, hook detection, subtitles, and music.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input / Output ──────────────────────────────────────────────────────────
    io = p.add_argument_group("Input / Output")
    io.add_argument("--input", "-i", required=False, metavar="FILE",
                    help="Source video file path.")
    io.add_argument("--output", "-o", metavar="FILE",
                    help="Output video file path. Default: output/<input>_edited.mp4")
    io.add_argument("--output-dir", metavar="DIR", default="output",
                    help="Output directory (default: output/)")
    io.add_argument("--batch", metavar="DIR",
                    help="Process all videos in a directory.")

    # ── Clip Mode ───────────────────────────────────────────────────────────────
    clip = p.add_argument_group("Clip Mode")
    clip.add_argument("--mode", choices=["short", "long"], default="long",
                      help="'short': extract a highlight clip. 'long': full video. (default: long)")
    clip.add_argument("--duration", type=float, default=60.0, metavar="SECONDS",
                      help="For --mode short: clip length in seconds. (default: 60)")
    clip.add_argument("--size", default="16:9", metavar="PRESET_OR_WxH",
                      help="Output size: 9:16 | 16:9 | 1:1 | 4:5 | custom WxH. (default: 16:9)")
    clip.add_argument("--fps", type=int, default=30,
                      help="Output frames per second. (default: 30)")

    # ── Silence Removal ──────────────────────────────────────────────────────────
    silence = p.add_argument_group("Silence Removal")
    silence.add_argument("--remove-silence", action="store_true",
                         help="Remove silent / dead-air segments.")
    silence.add_argument("--silence-thresh", type=float, default=-40.0, metavar="DB",
                         help="Silence threshold in dB (default: -40). Lower = more aggressive.")
    silence.add_argument("--min-silence-ms", type=int, default=500, metavar="MS",
                         help="Minimum silence duration to cut in ms (default: 500).")

    # ── Noise Reduction ───────────────────────────────────────────────────────────
    denoise = p.add_argument_group("Noise Reduction (DeepFilterNet)")
    denoise.add_argument("--denoise", action="store_true",
                         help="Remove background noise using DeepFilterNet before editing.")
    denoise.add_argument("--denoise-postfilter", action="store_true",
                         help="Enable DeepFilterNet post-filter for stronger noise suppression.")
    denoise.add_argument("--deep-filter-bin", metavar="PATH",
                         help="Path to deep-filter binary (default: tools/deep-filter/ or PATH).")
    denoise.add_argument("--denoise-workers", type=int, default=None, metavar="N",
                         help="Parallel denoise workers (default: up to 8 CPU cores).")

    # ── Hook Detection ───────────────────────────────────────────────────────────
    hook = p.add_argument_group("Hook Detection")
    hook.add_argument("--hook", action="store_true",
                      help="Auto-detect and extract the best hook segment.")
    hook.add_argument("--hook-duration", type=float, default=30.0, metavar="SECONDS",
                      help="Target hook clip length in seconds. (default: 30)")
    hook.add_argument("--hook-ai", action="store_true", default=True,
                      help="Use AI to score hook candidates (requires .env key). (default: on)")
    hook.add_argument("--no-hook-ai", dest="hook_ai", action="store_false",
                      help="Use audio-only hook detection (no AI).")

    # ── Subtitles ────────────────────────────────────────────────────────────────
    sub = p.add_argument_group("Subtitles")
    sub.add_argument("--subtitle", "--sub", action="store_true",
                     help="Generate and burn subtitles using Whisper.")
    sub.add_argument("--sub-style", default="default", metavar="STYLE",
                     choices=["default", "karaoke", "box", "gradient", "minimal"],
                     help="Subtitle style: default|karaoke|box|gradient|minimal. (default: default)")
    sub.add_argument("--sub-language", metavar="LANG",
                     help="Language code for Whisper e.g. 'th', 'en'. Auto-detect if omitted.")
    sub.add_argument("--whisper-model", default="base",
                     choices=["tiny", "base", "small", "medium", "large"],
                     help="Whisper model size (default: base).")
    sub.add_argument("--srt", metavar="FILE",
                     help="Use existing SRT file instead of running Whisper.")
    sub.add_argument("--transcribe-only", action="store_true",
                     help="Only run transcription (no video editing). Saves SRT/JSON/TXT.")

    # ── Background Music ─────────────────────────────────────────────────────────
    music = p.add_argument_group("Background Music")
    music.add_argument("--music", "-m", metavar="FILE",
                       help="Background music file path.")
    music.add_argument("--music-volume", type=float, default=0.15, metavar="RATIO",
                       help="BGM volume ratio 0.0-1.0 (default: 0.15).")
    music.add_argument("--no-ducking", dest="music_ducking", action="store_false", default=True,
                       help="Disable audio ducking (BGM won't lower during speech).")

    # ── Export Quality ────────────────────────────────────────────────────────────
    quality = p.add_argument_group("Export Quality")
    quality.add_argument("--crf", type=int, default=23, metavar="N",
                         help="H.264 CRF quality: 18=high, 23=default, 28=low.")
    quality.add_argument("--audio-bitrate", default="192k",
                         help="Audio encoding bitrate (default: 192k).")

    # ── Utilities ─────────────────────────────────────────────────────────────────
    util = p.add_argument_group("Utilities")
    util.add_argument("--info", action="store_true",
                      help="Print video info and exit.")
    util.add_argument("--list-sizes", action="store_true",
                      help="Print available size presets and exit.")
    util.add_argument("--check-env", action="store_true",
                      help="Check FFmpeg and AI provider availability.")
    util.add_argument("--config", metavar="FILE", default="config.yaml",
                      help="Path to YAML config file (default: config.yaml).")

    return p


def cmd_info(video_path: str):
    from src.clip_exporter import get_video_info
    info = get_video_info(video_path)
    print("\nVideo Info:")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")


def cmd_list_sizes():
    from src.clip_exporter import list_size_presets
    print("\nAvailable size presets:")
    for name, dims in list_size_presets().items():
        print(f"  {name:10s} → {dims}")
    print("  custom     → WxH (e.g. 1280x720)")


def cmd_check_env():
    from src.audio_extractor import check_ffmpeg
    from src.noise_reducer import check_deep_filter, default_denoise_workers, resolve_deep_filter_bin
    from src.ai_client import is_available, PROVIDER, OPENAI_KEY, ANTHROPIC_KEY, GOOGLE_KEY
    df_bin = resolve_deep_filter_bin()
    print("\nEnvironment Check:")
    print(f"  FFmpeg       : {'OK' if check_ffmpeg() else 'NOT FOUND — Install FFmpeg'}")
    print(f"  deep-filter  : {'OK' if check_deep_filter() else 'NOT FOUND — runs auto-download on first --denoise'}")
    if df_bin:
        print(f"                 ({df_bin})")
        print(f"  denoise CPU  : {default_denoise_workers()} parallel workers (deep-filter is CPU-only)")
    print(f"  AI provider  : {PROVIDER or '(auto-detect)'}")
    print(f"  OpenAI key   : {'SET' if OPENAI_KEY else 'not set'}")
    print(f"  Anthropic key: {'SET' if ANTHROPIC_KEY else 'not set'}")
    print(f"  Google key   : {'SET' if GOOGLE_KEY else 'not set'}")
    print(f"  AI available : {'YES' if is_available() else 'NO (audio-only hook detection)'}")


def cmd_transcribe_only(args):
    from src.audio_extractor import extract_audio
    from src.transcriber import transcribe

    audio_path = extract_audio(args.input)
    result = transcribe(
        audio_path,
        model_name=args.whisper_model,
        language=args.sub_language,
        formats=["srt", "json", "txt"],
    )
    print(f"\nTranscription complete.")
    print(f"  Language : {result['language']}")
    print(f"  Segments : {len(result['segments'])}")
    for key in ("srt_path", "json_path", "txt_path"):
        if key in result:
            print(f"  {key:10s}: {result[key]}")


def process_single(args) -> int:
    from src.editor import EditConfig, run_pipeline

    cfg = EditConfig(
        input_path=args.input,
        output_path=args.output or "",
        output_dir=args.output_dir,
        mode=args.mode,
        duration=args.duration,
        size=args.size,
        fps=args.fps,
        remove_silence=args.remove_silence,
        silence_thresh_db=args.silence_thresh,
        min_silence_ms=args.min_silence_ms,
        denoise=args.denoise,
        denoise_postfilter=args.denoise_postfilter,
        deep_filter_bin=args.deep_filter_bin,
        denoise_workers=args.denoise_workers,
        find_hook=args.hook,
        hook_duration=args.hook_duration,
        hook_ai=args.hook_ai,
        add_subtitle=args.subtitle,
        subtitle_style=args.sub_style,
        subtitle_language=args.sub_language,
        whisper_model=args.whisper_model,
        srt_path=args.srt,
        add_music=bool(args.music),
        music_path=args.music,
        music_volume=args.music_volume,
        music_ducking=args.music_ducking,
        crf=args.crf,
        audio_bitrate=args.audio_bitrate,
    )

    result = run_pipeline(cfg)
    if not result.success:
        print("\nErrors:")
        for err in result.errors:
            print(f"  - {err}")
        return 1
    return 0


def process_batch(batch_dir: str, args) -> int:
    from pathlib import Path
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
    videos = [f for f in Path(batch_dir).iterdir() if f.suffix.lower() in video_extensions]
    if not videos:
        print(f"No video files found in {batch_dir}")
        return 1

    print(f"\nBatch mode: {len(videos)} videos found in {batch_dir}")
    errors = 0
    for i, vid in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing: {vid.name}")
        args.input = str(vid)
        args.output = None
        rc = process_single(args)
        if rc != 0:
            errors += 1
    print(f"\nBatch complete: {len(videos) - errors}/{len(videos)} succeeded.")
    return 0 if errors == 0 else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_sizes:
        cmd_list_sizes()
        return 0

    if args.check_env:
        cmd_check_env()
        return 0

    if args.info:
        if not args.input:
            print("Error: --info requires --input")
            return 1
        cmd_info(args.input)
        return 0

    if args.batch:
        return process_batch(args.batch, args)

    if not args.input:
        parser.print_help()
        return 0

    if args.transcribe_only:
        cmd_transcribe_only(args)
        return 0

    return process_single(args)


if __name__ == "__main__":
    sys.exit(main())
