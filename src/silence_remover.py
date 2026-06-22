"""
silence_remover.py
Detects and removes silent / dead-air segments using pydub + parallel FFmpeg (GPU).
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.ffmpeg_utils import (
    default_parallel_workers,
    hwaccel_decode_args,
    probe_video,
    run_ffmpeg,
    video_encoder_args,
)


def detect_silence(
    audio_path: str,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 500,
    padding_ms: int = 100,
    max_gap_sec: float = 0.8,
) -> list[tuple[float, float]]:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    print(f"[silence_remover] Analyzing silence in {Path(audio_path).name} ...")
    audio = AudioSegment.from_wav(audio_path)

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
    )

    padding = padding_ms
    total_ms = len(audio)
    keep_ranges = []
    for start_ms, end_ms in nonsilent_ranges:
        start_ms = max(0, start_ms - padding)
        end_ms = min(total_ms, end_ms + padding)
        keep_ranges.append((start_ms / 1000.0, end_ms / 1000.0))

    merged = _merge_overlapping(keep_ranges)
    if max_gap_sec > 0:
        before = len(merged)
        merged = _merge_nearby_gaps(merged, max_gap_sec)
        if len(merged) < before:
            print(f"[silence_remover] Merged nearby gaps: {before} -> {len(merged)} segments")
    print(f"[silence_remover] Found {len(merged)} speech segments")
    return merged


def _merge_overlapping(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _merge_nearby_gaps(ranges: list[tuple[float, float]], max_gap_sec: float) -> list[tuple[float, float]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        gap = start - merged[-1][1]
        if gap <= max_gap_sec:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _extract_segment(
    video_path: str,
    start: float,
    end: float,
    out_path: str,
    crf: int,
) -> tuple[int, str | None]:
    duration = end - start
    if duration < 0.05:
        return -1, None

    encode = video_encoder_args(crf=crf, fast=True)
    audio = ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]

    for decode in (hwaccel_decode_args(), []):
        args = [
            *decode,
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            *encode,
            *audio,
            out_path,
        ]
        try:
            run_ffmpeg(args, label=f"segment@{start:.1f}s")
            return int(start * 1000), out_path
        except RuntimeError:
            if not decode:
                return int(start * 1000), "encode failed"
            continue
    return int(start * 1000), "encode failed"


def _ffmpeg_concat_files(inputs: list[str], output_path: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        for path in inputs:
            escaped = str(Path(path).resolve()).replace("'", "'\\''")
            tmp.write(f"file '{escaped}'\n")
        list_path = tmp.name

    try:
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path],
            label="silence_remover_concat",
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


def remove_silence(
    video_path: str,
    audio_path: str,
    output_path: str,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 500,
    padding_ms: int = 100,
    crf: int = 23,
    max_gap_sec: float = 0.8,
    workers: int | None = None,
) -> str:
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    src_mtime = Path(video_path).stat().st_mtime
    out_file = Path(output_path)
    if out_file.exists() and out_file.stat().st_mtime >= src_mtime:
        print(f"[silence_remover] Reusing cached: {output_path}")
        return output_path

    keep_ranges = detect_silence(
        audio_path,
        silence_thresh_db,
        min_silence_ms,
        padding_ms,
        max_gap_sec=max_gap_sec,
    )

    if not keep_ranges:
        print("[silence_remover] No speech detected - skipping silence removal.")
        return video_path

    original_duration = probe_video(video_path)["duration"]
    worker_count = workers or default_parallel_workers()
    temp_dir = out_file.parent / f"{out_file.stem}_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[silence_remover] Extracting {len(keep_ranges)} segments "
        f"({worker_count} parallel workers, NVENC/CUDA if available) ..."
    )

    jobs: list[tuple[int, float, float, str]] = []
    for i, (start, end) in enumerate(keep_ranges):
        seg_path = str(temp_dir / f"seg_{i:04d}.mp4")
        jobs.append((i, start, end, seg_path))

    results: list[tuple[int, str]] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_extract_segment, video_path, start, end, seg_path, crf): idx
            for idx, start, end, seg_path in jobs
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            _, start, _, _ = jobs[idx]
            sort_key, result = future.result()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"[silence_remover] Progress: {done}/{len(jobs)} segments")
            if result is None:
                continue
            if result.endswith(".mp4"):
                results.append((sort_key, result))
            else:
                errors.append(f"seg {idx} @ {start:.1f}s: {result}")

    if not results:
        raise RuntimeError(f"[silence_remover] All segment extractions failed. First error: {errors[:1]}")

    results.sort(key=lambda x: x[0])
    segment_paths = [path for _, path in results]

    print(f"[silence_remover] Concatenating {len(segment_paths)} segments ...")
    _ffmpeg_concat_files(segment_paths, output_path)

    for path in segment_paths:
        Path(path).unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    new_duration = probe_video(output_path)["duration"]
    saved_sec = max(0.0, original_duration - new_duration)
    print(f"[silence_remover] Removed ~{saved_sec:.1f}s of silence.")
    if errors:
        print(f"[silence_remover] Warning: {len(errors)} segments failed (skipped).")
    return output_path


def get_silence_stats(audio_path: str, silence_thresh_db: float = -40.0, min_silence_ms: int = 500) -> dict:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    audio = AudioSegment.from_wav(audio_path)
    total_sec = len(audio) / 1000.0

    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_ms, silence_thresh=silence_thresh_db)
    speech_sec = sum((e - s) / 1000.0 for s, e in nonsilent)
    silence_sec = total_sec - speech_sec

    return {
        "total_duration": round(total_sec, 2),
        "speech_duration": round(speech_sec, 2),
        "silence_duration": round(silence_sec, 2),
        "silence_ratio": round(silence_sec / total_sec, 3) if total_sec > 0 else 0,
        "speech_segments": len(nonsilent),
    }
