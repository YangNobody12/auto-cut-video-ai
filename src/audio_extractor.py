"""
audio_extractor.py
Extracts audio from a video file once to a shared temp WAV.
All downstream modules (Whisper, silence remover, hook detector) reuse this file.
"""
import os
import subprocess
from pathlib import Path


def extract_audio(
    video_path: str,
    output_path: str | None = None,
    sample_rate: int = 16000,
    force: bool = False,
) -> str:
    """
    Extract audio from video to a mono WAV file using FFmpeg.

    Args:
        video_path: Path to the source video file.
        output_path: Destination WAV path. Defaults to temp/<video_stem>.wav
        sample_rate: Sample rate in Hz. 16000 is optimal for Whisper + librosa.

    Returns:
        Absolute path to the extracted WAV file.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        temp_dir = video_path.parent.parent / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / f"{video_path.stem}.wav"

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        if output_path.stat().st_mtime >= video_path.stat().st_mtime:
            print(f"[audio_extractor] Reusing cached audio: {output_path}")
            return str(output_path)
        print(f"[audio_extractor] Source video newer than cache, re-extracting ...")

    print(f"[audio_extractor] Extracting audio from {video_path.name} ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                       # no video
        "-ac", "1",                  # mono
        "-ar", str(sample_rate),     # sample rate
        "-acodec", "pcm_s16le",      # 16-bit PCM
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")

    print(f"[audio_extractor] Saved to: {output_path}")
    return str(output_path)


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_extractor.py <video_file>")
        sys.exit(1)
    out = extract_audio(sys.argv[1])
    print(f"Audio extracted to: {out}")
