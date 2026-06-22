"""
transcriber.py
Transcribes audio using faster-whisper and outputs SRT, VTT, or JSON with timestamps.
"""
import json
import os
import sys
from pathlib import Path


WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
MODEL_ALIASES = {"large": "large-v3"}


def _resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def _setup_cuda_dll_paths() -> None:
    """Add NVIDIA pip package DLL dirs so ctranslate2 can load cublas/cudnn on Windows."""
    if sys.platform != "win32":
        return

    try:
        import site
    except ImportError:
        return

    search_roots: list[Path] = []
    for entry in site.getsitepackages() + [site.getusersitepackages()]:
        if entry:
            search_roots.append(Path(entry))

    seen: set[str] = set()
    for root in search_roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for bin_dir in nvidia_root.rglob("bin"):
            if not bin_dir.is_dir():
                continue
            key = str(bin_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            os.add_dll_directory(key)
            os.environ["PATH"] = key + os.pathsep + os.environ.get("PATH", "")


def _create_model(model_name: str, device: str | None = None, compute_type: str | None = None):
    """Load faster-whisper model on the requested or auto-detected device."""
    _setup_cuda_dll_paths()
    from faster_whisper import WhisperModel

    resolved_model = _resolve_model_name(model_name)

    if device and compute_type:
        print(f"[transcriber] Loading faster-whisper model '{resolved_model}' ({device}/{compute_type}) ...")
        return WhisperModel(resolved_model, device=device, compute_type=compute_type)

    candidates: list[tuple[str, str]] = []
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            candidates.append(("cuda", "float16"))
    except Exception:
        pass
    candidates.append(("cpu", "int8"))

    last_error: Exception | None = None
    for dev, ctype in candidates:
        try:
            print(f"[transcriber] Loading faster-whisper model '{resolved_model}' ({dev}/{ctype}) ...")
            return WhisperModel(resolved_model, device=dev, compute_type=ctype)
        except Exception as exc:
            last_error = exc
            print(f"[transcriber] {dev} unavailable: {exc}")

    raise RuntimeError(f"Could not load faster-whisper model '{resolved_model}'") from last_error


def _run_transcription(
    model,
    audio_path: Path,
    language: str | None,
    *,
    vad_filter: bool = True,
    beam_size: int = 1,
):
    seg_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        vad_filter=vad_filter,
        no_speech_threshold=0.4 if not vad_filter else 0.6,
    )

    segments: list[dict] = []
    parts: list[str] = []
    for seg in seg_gen:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": seg.start, "end": seg.end, "text": text})
        parts.append(text)

    full_text = " ".join(parts).strip()
    detected_language = info.language or "unknown"
    return segments, full_text, detected_language


def _transcribe_meta_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{stem}.transcribe.meta.json"


def _transcribe_params_match(meta: dict, model_name: str, language: str | None, vad_filter: bool, beam_size: int) -> bool:
    return (
        meta.get("model_name") == model_name
        and meta.get("language") == language
        and meta.get("vad_filter") == vad_filter
        and meta.get("beam_size") == beam_size
    )


def _write_transcribe_meta(
    meta_path: Path,
    audio_path: Path,
    model_name: str,
    language: str | None,
    vad_filter: bool,
    beam_size: int,
) -> None:
    meta_path.write_text(
        json.dumps(
            {
                "audio_path": str(audio_path),
                "audio_mtime_ns": audio_path.stat().st_mtime_ns,
                "model_name": model_name,
                "language": language,
                "vad_filter": vad_filter,
                "beam_size": beam_size,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_cached_transcription(
    audio_path: Path,
    output_dir: Path,
    formats: list[str],
    model_name: str,
    language: str | None,
    vad_filter: bool,
    beam_size: int,
) -> dict | None:
    stem = audio_path.stem
    json_path = output_dir / f"{stem}.json"
    if not json_path.exists():
        return None

    if json_path.stat().st_mtime < audio_path.stat().st_mtime:
        return None

    meta_path = _transcribe_meta_path(output_dir, stem)
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta and not _transcribe_params_match(meta, model_name, language, vad_filter, beam_size):
            return None

    segments = load_segments_from_json(str(json_path))
    if not segments:
        return None

    saved: dict[str, str] = {"json_path": str(json_path)}
    srt_path = output_dir / f"{stem}.srt"
    if "srt" in formats and srt_path.exists():
        saved["srt_path"] = str(srt_path)
    vtt_path = output_dir / f"{stem}.vtt"
    if "vtt" in formats and vtt_path.exists():
        saved["vtt_path"] = str(vtt_path)
    txt_path = output_dir / f"{stem}.txt"
    if "txt" in formats and txt_path.exists():
        saved["txt_path"] = str(txt_path)

    full_text = " ".join(seg["text"] for seg in segments if seg.get("text")).strip()
    print(f"[transcriber] Reusing cached transcript: {json_path} ({len(segments)} segments)")
    return {
        "segments": segments,
        "language": meta.get("language") or language or "unknown",
        "text": full_text,
        **saved,
    }


def transcribe(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
    output_dir: str | None = None,
    formats: list[str] | None = None,
    vad_filter: bool = True,
    beam_size: int = 1,
    force: bool = False,
) -> dict:
    """
    Transcribe audio file with faster-whisper.

    Args:
        audio_path: Path to WAV/MP3/etc audio file.
        model_name: Whisper model size (tiny/base/small/medium/large).
        language: ISO language code e.g. 'th', 'en'. None = auto-detect.
        output_dir: Where to save SRT/VTT/JSON. Defaults to same dir as audio.
        formats: List of output formats. Supported: 'srt', 'vtt', 'json', 'txt'.

    Returns:
        Dict with keys:
            - segments: list of {start, end, text}
            - language: detected language
            - text: full transcript text
            - srt_path / vtt_path / json_path / txt_path: saved file paths
    """
    if formats is None:
        formats = ["srt", "json"]

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if output_dir is None:
        output_dir = audio_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        cached = _load_cached_transcription(
            audio_path, output_dir, formats, model_name, language, vad_filter, beam_size
        )
        if cached:
            return cached

    model = _create_model(model_name)

    print(f"[transcriber] Transcribing {audio_path.name} ...")
    try:
        segments, full_text, detected_language = _run_transcription(
            model, audio_path, language, vad_filter=vad_filter, beam_size=beam_size
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "cuda" in msg or "cublas" in msg or "cudnn" in msg:
            print(f"[transcriber] Accelerator inference failed, retrying on CPU: {exc}")
            model = _create_model(model_name, device="cpu", compute_type="int8")
            segments, full_text, detected_language = _run_transcription(
                model, audio_path, language, vad_filter=vad_filter, beam_size=beam_size
            )
        else:
            raise

    saved = {}
    stem = audio_path.stem

    if "srt" in formats:
        srt_path = output_dir / f"{stem}.srt"
        _write_srt(segments, srt_path)
        saved["srt_path"] = str(srt_path)
        print(f"[transcriber] SRT saved: {srt_path}")

    if "vtt" in formats:
        vtt_path = output_dir / f"{stem}.vtt"
        _write_vtt(segments, vtt_path)
        saved["vtt_path"] = str(vtt_path)
        print(f"[transcriber] VTT saved: {vtt_path}")

    if "json" in formats:
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        saved["json_path"] = str(json_path)
        print(f"[transcriber] JSON saved: {json_path}")

    if "txt" in formats:
        txt_path = output_dir / f"{stem}.txt"
        txt_path.write_text(full_text, encoding="utf-8")
        saved["txt_path"] = str(txt_path)
        print(f"[transcriber] TXT saved: {txt_path}")

    _write_transcribe_meta(
        _transcribe_meta_path(output_dir, stem),
        audio_path,
        model_name,
        language,
        vad_filter,
        beam_size,
    )

    return {
        "segments": segments,
        "language": detected_language,
        "text": full_text,
        **saved,
    }


def load_segments_from_json(json_path: str) -> list[dict]:
    """Load previously saved transcript segments."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _write_srt(segments: list[dict], path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(seg['start'])} --> {_format_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_vtt(segments: list[dict], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_format_vtt_time(seg['start'])} --> {_format_vtt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file> [model] [language]")
        sys.exit(1)
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    lang = sys.argv[3] if len(sys.argv) > 3 else None
    result = transcribe(sys.argv[1], model_name=model, language=lang, formats=["srt", "json", "txt"])
    print(f"Language detected: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
