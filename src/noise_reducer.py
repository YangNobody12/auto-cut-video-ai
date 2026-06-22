"""
noise_reducer.py
Background noise removal using DeepFilterNet's deep-filter CLI.

Long files are split into chunks and processed in parallel across CPU cores.
Requires 48 kHz mono WAV input. See:
https://github.com/Rikorose/DeepFilterNet/releases
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.audio_extractor import extract_audio
from src.ffmpeg_utils import run_ffmpeg

DEEP_FILTER_VERSION = "0.5.6"
_RELEASE_BASE = (
    f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{DEEP_FILTER_VERSION}"
)
_DOWNLOAD_URLS = {
    "Windows": f"{_RELEASE_BASE}/deep-filter-{DEEP_FILTER_VERSION}-x86_64-pc-windows-msvc.exe",
    "Linux": f"{_RELEASE_BASE}/deep-filter-{DEEP_FILTER_VERSION}-x86_64-unknown-linux-gnu",
    "Darwin": f"{_RELEASE_BASE}/deep-filter-{DEEP_FILTER_VERSION}-aarch64-apple-darwin",
}

DEFAULT_CHUNK_SEC = 60.0
PARALLEL_THRESHOLD_SEC = 30.0


def default_denoise_workers() -> int:
    """Use most CPU cores; each worker runs an isolated deep-filter process."""
    cores = os.cpu_count() or 4
    return max(2, min(cores, 12))


def _single_thread_env() -> dict[str, str]:
    """Prevent ONNX/OpenMP from spawning extra threads inside each worker."""
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ORT_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def _project_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "tools" / "deep-filter"


def _bundled_binary_name() -> str:
    return "deep-filter.exe" if platform.system() == "Windows" else "deep-filter"


def resolve_deep_filter_bin(custom_path: str | None = None) -> Path | None:
    """Return path to deep-filter binary if found."""
    candidates: list[Path] = []

    if custom_path:
        candidates.append(Path(custom_path))
    env_bin = os.environ.get("DEEP_FILTER_BIN")
    if env_bin:
        candidates.append(Path(env_bin))

    bundled = _project_tools_dir() / _bundled_binary_name()
    candidates.append(bundled)

    for name in ("deep-filter", "deep-filter.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def check_deep_filter(custom_path: str | None = None) -> bool:
    return resolve_deep_filter_bin(custom_path) is not None


def download_deep_filter(dest_dir: Path | None = None) -> Path:
    """Download the pre-built deep-filter binary for this OS."""
    system = platform.system()
    url = _DOWNLOAD_URLS.get(system)
    if not url:
        raise RuntimeError(
            f"[noise_reducer] No pre-built deep-filter binary for {system}. "
            "Download manually from https://github.com/Rikorose/DeepFilterNet/releases"
        )

    dest_dir = dest_dir or _project_tools_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _bundled_binary_name()

    print(f"[noise_reducer] Downloading deep-filter {DEEP_FILTER_VERSION} ...")
    urllib.request.urlretrieve(url, dest)

    if system != "Windows":
        dest.chmod(dest.stat().st_mode | 0o111)

    print(f"[noise_reducer] Saved to: {dest}")
    return dest


def ensure_deep_filter(custom_path: str | None = None, auto_download: bool = True) -> Path:
    """Locate deep-filter, optionally downloading it on first use."""
    found = resolve_deep_filter_bin(custom_path)
    if found:
        return found

    if auto_download and platform.system() in _DOWNLOAD_URLS:
        return download_deep_filter()

    raise FileNotFoundError(
        "[noise_reducer] deep-filter not found. Install from "
        "https://github.com/Rikorose/DeepFilterNet/releases or set DEEP_FILTER_BIN."
    )


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        if not rate:
            return 0.0
        return wf.getnframes() / float(rate)


def _run_deep_filter(
    binary: Path,
    input_wav: Path,
    out_dir: Path,
    postfilter: bool,
    compensate_delay: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-o", str(out_dir), str(input_wav)]
    if postfilter:
        cmd.append("--pf")
    if compensate_delay:
        cmd.append("-D")

    result = subprocess.run(cmd, capture_output=True, text=True, env=_single_thread_env())
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"[noise_reducer] deep-filter failed on {input_wav.name}:\n{detail}")

    enhanced = out_dir / input_wav.name
    if enhanced.exists():
        return enhanced

    matches = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise RuntimeError(f"[noise_reducer] deep-filter produced no output in {out_dir}")
    return matches[0]


def _split_wav_chunks(input_wav: Path, chunks_dir: Path, chunk_sec: float) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    marker = chunks_dir / ".source"
    marker_key = f"{input_wav.stat().st_mtime_ns}:{chunk_sec}"
    existing = sorted(chunks_dir.glob("chunk_*.wav"))
    if existing and marker.exists() and marker.read_text(encoding="utf-8").strip() == marker_key:
        print(f"[noise_reducer] Reusing {len(existing)} audio chunks")
        return existing

    for old in chunks_dir.glob("chunk_*.wav"):
        old.unlink()

    pattern = str(chunks_dir / "chunk_%04d.wav")
    run_ffmpeg(
        [
            "-threads", "0",
            "-i", str(input_wav),
            "-f", "segment",
            "-segment_time", f"{chunk_sec:.3f}",
            "-ar", "48000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            pattern,
        ],
        label="noise_reducer/split",
    )
    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError(f"[noise_reducer] Failed to split {input_wav.name}")
    marker.write_text(marker_key, encoding="utf-8")
    return chunks


def _concat_wav_files(inputs: list[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        list_path = tmp.name
        for path in inputs:
            escaped = str(path.resolve()).replace("'", "'\\''")
            tmp.write(f"file '{escaped}'\n")

    try:
        run_ffmpeg(
            [
                "-threads", "0",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-ar", "48000",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                str(output_path),
            ],
            label="noise_reducer/concat",
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


def _denoise_chunk_task(
    chunk_path: Path,
    out_root: Path,
    binary: Path,
    postfilter: bool,
    compensate_delay: bool,
) -> tuple[int, Path, bool]:
    chunk_out = out_root / chunk_path.stem
    enhanced = chunk_out / chunk_path.name
    idx = int(chunk_path.stem.rsplit("_", 1)[-1])

    if enhanced.exists() and enhanced.stat().st_mtime >= chunk_path.stat().st_mtime:
        return idx, enhanced, True

    enhanced = _run_deep_filter(binary, chunk_path, chunk_out, postfilter, compensate_delay)
    return idx, enhanced, False


def _denoise_wav_parallel(
    input_path: Path,
    enhanced_path: Path,
    out_dir: Path,
    binary: Path,
    postfilter: bool,
    compensate_delay: bool,
    workers: int,
    chunk_sec: float,
) -> None:
    chunks_dir = out_dir / "_chunks"
    duration = _wav_duration_sec(input_path)
    chunks = _split_wav_chunks(input_path, chunks_dir, chunk_sec)
    chunk_out_root = out_dir / "_chunk_out"

    print(
        f"[noise_reducer] Parallel denoise: {len(chunks)} chunks x ~{chunk_sec:.0f}s, "
        f"{workers} workers (~{duration / 60:.0f} min audio) ..."
    )
    t0 = time.time()
    results: dict[int, Path] = {}
    errors: list[str] = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _denoise_chunk_task,
                chunk,
                chunk_out_root,
                binary,
                postfilter,
                compensate_delay,
            ): chunk
            for chunk in chunks
        }
        done = 0
        for future in as_completed(futures):
            chunk = futures[future]
            done += 1
            try:
                idx, path, was_cached = future.result()
                results[idx] = path
                if was_cached:
                    skipped += 1
            except Exception as exc:
                errors.append(f"{chunk.name}: {exc}")

            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(chunks) - done) / rate if rate > 0 else 0
            print(
                f"[noise_reducer] Progress: {done}/{len(chunks)} chunks "
                f"({skipped} cached) — ETA {remaining:.0f}s"
            )

    if errors:
        raise RuntimeError(
            f"[noise_reducer] {len(errors)} chunk(s) failed. First error: {errors[0]}"
        )

    ordered = [results[i] for i in sorted(results)]
    _concat_wav_files(ordered, enhanced_path)
    elapsed = time.time() - t0
    print(f"[noise_reducer] Parallel denoise finished in {elapsed:.1f}s")


def denoise_wav(
    input_wav: str,
    output_dir: str | None = None,
    postfilter: bool = False,
    compensate_delay: bool = True,
    deep_filter_bin: str | None = None,
    workers: int | None = None,
    chunk_sec: float = DEFAULT_CHUNK_SEC,
) -> str:
    """
    Run DeepFilterNet on a 48 kHz WAV file.

    Returns:
        Path to the enhanced WAV (same filename inside output_dir).
    """
    input_path = Path(input_wav).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"WAV not found: {input_path}")

    out_dir = Path(output_dir or input_path.parent / "df_out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_path = out_dir / input_path.name

    if enhanced_path.exists() and enhanced_path.stat().st_mtime >= input_path.stat().st_mtime:
        print(f"[noise_reducer] Reusing enhanced audio: {enhanced_path}")
        return str(enhanced_path)

    binary = ensure_deep_filter(deep_filter_bin)
    duration = _wav_duration_sec(input_path)
    worker_count = workers if workers is not None else default_denoise_workers()

    if duration <= PARALLEL_THRESHOLD_SEC or worker_count <= 1:
        print(f"[noise_reducer] Denoising {input_path.name} ({duration:.1f}s) ...")
        t0 = time.time()
        enhanced = _run_deep_filter(binary, input_path, out_dir, postfilter, compensate_delay)
        if enhanced != enhanced_path:
            shutil.move(str(enhanced), str(enhanced_path))
        print(f"[noise_reducer] Denoise finished in {time.time() - t0:.1f}s")
    else:
        _denoise_wav_parallel(
            input_path,
            enhanced_path,
            out_dir,
            binary,
            postfilter,
            compensate_delay,
            worker_count,
            chunk_sec,
        )

    print(f"[noise_reducer] Enhanced audio saved: {enhanced_path}")
    return str(enhanced_path)


def denoise_video(
    video_path: str,
    output_path: str,
    temp_dir: Path | None = None,
    postfilter: bool = False,
    deep_filter_bin: str | None = None,
    workers: int | None = None,
    chunk_sec: float = DEFAULT_CHUNK_SEC,
    force: bool = False,
) -> str:
    """
    Extract 48 kHz audio, denoise with DeepFilterNet, and mux clean audio back into the video.

    Returns:
        Path to the output video with denoised audio.
    """
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and output_path.exists()
        and output_path.stat().st_mtime >= video_path.stat().st_mtime
    ):
        print(f"[noise_reducer] Reusing denoised video: {output_path}")
        return str(output_path)

    temp_dir = temp_dir or Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    wav_48k = temp_dir / f"{video_path.stem}_48k.wav"
    extract_audio(str(video_path), str(wav_48k), sample_rate=48000, force=force)

    df_out_dir = temp_dir / f"{video_path.stem}_df"
    enhanced_wav = denoise_wav(
        str(wav_48k),
        output_dir=str(df_out_dir),
        postfilter=postfilter,
        deep_filter_bin=deep_filter_bin,
        workers=workers,
        chunk_sec=chunk_sec,
    )

    print(f"[noise_reducer] Muxing denoised audio into {output_path.name} ...")
    run_ffmpeg(
        [
            "-threads", "0",
            "-i", str(video_path),
            "-i", enhanced_wav,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            str(output_path),
        ],
        label="noise_reducer/mux",
    )

    print(f"[noise_reducer] Denoised video saved: {output_path}")
    return str(output_path)
