"""Shared FFmpeg helpers for fast video processing."""
import json
import os
import shutil
import subprocess
from pathlib import Path

_NVENC_AVAILABLE: bool | None = None


def run_ffmpeg(args: list[str], label: str = "ffmpeg", cwd: str | Path | None = None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stderr.strip()}")


def has_nvenc() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        _NVENC_AVAILABLE = "h264_nvenc" in result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def hwaccel_decode_args() -> list[str]:
    if has_nvenc():
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    return []


def video_encoder_args(crf: int = 23, preset: str = "veryfast", fast: bool = False) -> list[str]:
    if has_nvenc():
        nvenc_preset = "p1" if fast else "p4"
        return ["-c:v", "h264_nvenc", "-preset", nvenc_preset, "-cq", str(crf)]
    x264_preset = "ultrafast" if fast else preset
    return ["-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf)]


def default_parallel_workers() -> int:
    return max(2, min(6, (os.cpu_count() or 4) // 2))


def escape_filter_path(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    return resolved.replace("\\", "/").replace(":", "\\:")


def probe_video(path: str | Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    fps = 30.0
    rate = stream.get("r_frame_rate", "30/1")
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        if float(den):
            fps = float(num) / float(den)
    return {"width": width, "height": height, "duration": duration, "fps": fps}


def copy_or_reencode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    crf: int = 23,
    audio_bitrate: str = "192k",
) -> str:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        return str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    shutil.copy2(input_path, output_path)
    return str(output_path)
