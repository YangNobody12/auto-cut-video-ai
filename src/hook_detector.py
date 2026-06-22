"""
hook_detector.py
Two-stage hook detection:
  Stage 1: Audio energy analysis (always runs, no API key needed)
  Stage 2: AI scoring via ai_client (runs if API key configured)
"""
from pathlib import Path


def _compute_energy_scores(audio_path: str, window_sec: float, hop_sec: float) -> list[dict]:
    """
    Slide a window over the audio and compute RMS energy for each position.
    Returns list of {start, end, energy_score}.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    total_sec = len(y) / sr

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    scores = []
    pos = 0
    while pos + window_samples <= len(y):
        chunk = y[pos : pos + window_samples]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        start_sec = pos / sr
        scores.append({
            "start": round(start_sec, 2),
            "end": round(start_sec + window_sec, 2),
            "energy_score": rms,
        })
        pos += hop_samples

    if not scores:
        return scores

    max_e = max(s["energy_score"] for s in scores)
    if max_e > 0:
        for s in scores:
            s["energy_score"] = round(s["energy_score"] / max_e, 4)

    return scores


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    span = min(a_end - a_start, b_end - b_start)
    return overlap / span if span > 0 else 0.0


def _score_segments_with_transcript(
    energy_scores: list[dict],
    segments: list[dict],
) -> list[dict]:
    """
    Boost energy score by speech density (words per second in window).
    """
    import numpy as np

    result = []
    for es in energy_scores:
        ws = es["start"]
        we = es["end"]
        win_dur = we - ws

        words_in_window = 0
        for seg in segments:
            overlap = _overlap_ratio(ws, we, seg["start"], seg["end"])
            if overlap > 0:
                seg_words = len(seg["text"].split())
                words_in_window += seg_words * overlap

        speech_density = words_in_window / win_dur if win_dur > 0 else 0
        combined = 0.6 * es["energy_score"] + 0.4 * min(speech_density / 3.0, 1.0)
        result.append({**es, "speech_density": round(speech_density, 3), "combined_score": round(combined, 4)})

    return result


def detect_hook(
    audio_path: str,
    segments: list[dict] | None = None,
    target_duration: float = 30.0,
    top_k: int = 5,
    use_ai: bool = True,
) -> dict:
    """
    Find the best hook segment in a video.

    Args:
        audio_path: Path to extracted audio WAV.
        segments: Whisper transcript segments [{start, end, text}].
        target_duration: Desired hook length in seconds.
        top_k: How many candidate windows to send to AI for scoring.
        use_ai: Whether to call AI for scoring (requires .env key).

    Returns:
        {
            "start": float,
            "end": float,
            "score": float,
            "reason": str,
            "method": "ai" | "audio"
        }
    """
    print(f"[hook_detector] Scanning audio for hook candidates (window={target_duration:.0f}s) ...")

    hop_sec = max(1.0, target_duration / 10)
    energy_scores = _compute_energy_scores(audio_path, target_duration, hop_sec)

    if not energy_scores:
        return {"start": 0.0, "end": target_duration, "score": 0.0, "reason": "No audio data", "method": "audio"}

    if segments:
        scored = _score_segments_with_transcript(energy_scores, segments)
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
    else:
        scored = sorted(energy_scores, key=lambda x: x["energy_score"], reverse=True)

    top_candidates = scored[:top_k]

    if use_ai and segments:
        from src.ai_client import score_hook, is_available
        if is_available():
            print(f"[hook_detector] Sending {len(segments)} transcript segments to AI ...")
            ai_result = score_hook(segments, target_duration=target_duration)
            if ai_result:
                print(f"[hook_detector] AI hook: {ai_result['best_start']:.1f}s - {ai_result['best_end']:.1f}s (score={ai_result['score']:.2f})")
                print(f"[hook_detector] Reason: {ai_result['reason']}")
                return {
                    "start": float(ai_result["best_start"]),
                    "end": float(ai_result["best_end"]),
                    "score": float(ai_result["score"]),
                    "reason": ai_result["reason"],
                    "method": "ai",
                }

    best = top_candidates[0]
    score_val = best.get("combined_score", best.get("energy_score", 0))
    print(f"[hook_detector] Audio-based hook: {best['start']:.1f}s - {best['end']:.1f}s (score={score_val:.2f})")
    return {
        "start": best["start"],
        "end": best["end"],
        "score": score_val,
        "reason": "Highest audio energy + speech density segment",
        "method": "audio",
    }


def extract_hook_clip(
    video_path: str,
    hook_result: dict,
    output_path: str,
) -> str:
    """
    Cut the hook segment from the video and save it.

    Args:
        video_path: Source video path.
        hook_result: Output from detect_hook().
        output_path: Destination path for the hook clip.

    Returns:
        Path to saved hook clip.
    """
    from moviepy import VideoFileClip

    start = hook_result["start"]
    end = hook_result["end"]

    print(f"[hook_detector] Extracting hook clip {start:.1f}s - {end:.1f}s ...")
    video = VideoFileClip(video_path)
    end = min(end, video.duration)
    clip = video.subclipped(start, end)

    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()
    video.close()
    print(f"[hook_detector] Hook clip saved: {output_path}")
    return output_path
