"""
hook_detector.py
Finds viral-style hook clips using transcript candidates + heuristics + AI ranking.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.ffmpeg_utils import run_ffmpeg, video_encoder_args

_BORING_START_RE = re.compile(
    r"^\s*(?:"
    r"สวัสดี|หวัดดี|hello|hi|hey|ok|โอเค|เอ่อ|um+|uh+|well|so|"
    r"วันนี้|today|ต่อไป|next|แล้วก็|and then|"
    r"ขอบคุณ|thank you|welcome"
    r")\b",
    re.IGNORECASE,
)

_HOOK_KEYWORDS = re.compile(
    r"(?:"
    r"ทำไม|อย่า|เดี๋ยว|ลับ|เคล็ด|ฟรี|พลาด|โหด|แรง| shocking|secret|never|best|worst|"
    r"how to|why |what if|did you|must|stop|warning|mistake|"
    r"รู้ไหม|คุณต้อง|ใคร|กี่|เท่าไห|ล้าน|พัน|\?|!"
    r")",
    re.IGNORECASE,
)


def _compute_energy_scores(audio_path: str, window_sec: float, hop_sec: float) -> list[dict]:
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=None, mono=True)
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


def _text_is_substantial(text: str) -> bool:
    text = text.strip()
    if len(text) >= 30:
        return True
    return len(text.split()) >= 6


def _merge_segment_text(segments: list[dict], start_idx: int, end_idx: int) -> str:
    return " ".join(seg["text"].strip() for seg in segments[start_idx : end_idx + 1] if seg.get("text"))


def _heuristic_score(text: str, start: float, end: float, energy: float = 0.5) -> float:
    text = text.strip()
    if not text:
        return 0.0

    score = 0.25 + 0.25 * energy
    words = text.split()
    wps = max(len(words), len(text) / 6) / max(0.5, end - start)
    if 2.0 <= wps <= 4.5:
        score += 0.15
    elif wps > 5.0:
        score += 0.05
    else:
        score -= 0.1

    first_line = text.split("\n")[0][:120]
    if _BORING_START_RE.search(first_line):
        score -= 0.35
    if _HOOK_KEYWORDS.search(first_line[:80]):
        score += 0.25
    if "?" in first_line[:100]:
        score += 0.12
    if "!" in first_line[:100]:
        score += 0.08
    if re.search(r"\d+", first_line[:80]):
        score += 0.08
    if len(first_line) < 12:
        score -= 0.15

    unique_ratio = len(set(words)) / max(1, len(words))
    if unique_ratio < 0.35:
        score -= 0.25

    return round(max(0.0, min(1.0, score)), 4)


def _snap_to_segments(start: float, end: float, segments: list[dict], target_duration: float) -> tuple[float, float]:
    if not segments:
        return start, end

    video_end = segments[-1]["end"]
    start = max(0.0, start)
    end = min(end, video_end)

    start_seg = min(segments, key=lambda s: abs(s["start"] - start))
    start = start_seg["start"]
    end = min(start + target_duration, video_end)

    if end - start < target_duration * 0.6:
        end = min(start + target_duration, video_end)

    return round(start, 2), round(end, 2)


def build_hook_candidates(
    segments: list[dict],
    target_duration: float,
    audio_path: str | None = None,
    max_candidates: int = 12,
) -> list[dict]:
    """Build clip candidates from transcript windows + audio energy peaks."""
    if not segments:
        return []

    video_end = segments[-1]["end"]
    min_dur = max(8.0, target_duration * 0.65)
    candidates: list[dict] = []
    seen_starts: set[int] = set()

    for i in range(len(segments)):
        if i in seen_starts:
            continue
        start = segments[i]["start"]
        end = start
        j = i
        while j < len(segments) and end - start < target_duration:
            end = segments[j]["end"]
            j += 1

        duration = end - start
        if duration < min_dur:
            continue

        clip_end = min(start + target_duration, end)
        text = _merge_segment_text(segments, i, j - 1)
        if not _text_is_substantial(text):
            continue

        energy = 0.5
        if audio_path:
            hop = max(1.0, target_duration / 10)
            for es in _compute_energy_scores(audio_path, target_duration, hop):
                if _overlap(es["start"], es["end"], start, clip_end) > 0.3:
                    energy = max(energy, es.get("energy_score", 0.5))

        h_score = _heuristic_score(text, start, clip_end, energy)
        candidates.append({
            "id": len(candidates) + 1,
            "start": round(start, 2),
            "end": round(clip_end, 2),
            "text": text,
            "heuristic_score": h_score,
        })
        seen_starts.add(i)

        if len(candidates) >= max_candidates * 2:
            break

    if audio_path and len(candidates) < max_candidates:
        hop = max(1.0, target_duration / 8)
        for es in _compute_energy_scores(audio_path, target_duration, hop)[:20]:
            if any(_overlap(es["start"], es["end"], c["start"], c["end"]) > 0.5 for c in candidates):
                continue
            nearest = min(segments, key=lambda s: abs(s["start"] - es["start"]))
            idx = segments.index(nearest)
            start = nearest["start"]
            end = start
            j = idx
            while j < len(segments) and end - start < target_duration:
                end = segments[j]["end"]
                j += 1
            if end - start < min_dur:
                continue
            clip_end = min(start + target_duration, end)
            text = _merge_segment_text(segments, idx, j - 1)
            if not _text_is_substantial(text):
                continue
            candidates.append({
                "id": len(candidates) + 1,
                "start": round(start, 2),
                "end": round(clip_end, 2),
                "text": text,
                "heuristic_score": _heuristic_score(text, start, clip_end, es.get("energy_score", 0.5)),
            })

    candidates.sort(key=lambda c: c["heuristic_score"], reverse=True)
    top = candidates[:max_candidates]
    for n, c in enumerate(top, 1):
        c["id"] = n
    return top


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    span = min(a_end - a_start, b_end - b_start)
    return overlap / span if span > 0 else 0.0


def detect_hook(
    audio_path: str,
    segments: list[dict] | None = None,
    target_duration: float = 30.0,
    top_k: int = 8,
    use_ai: bool = True,
) -> dict:
    """
    Find the best hook segment in a video.

    Returns dict with start, end, score, reason, method, preview_text.
    """
    print(f"[hook_detector] Building hook candidates (~{target_duration:.0f}s each) ...")

    if not segments:
        hop_sec = max(1.0, target_duration / 10)
        energy_scores = _compute_energy_scores(audio_path, target_duration, hop_sec)
        if not energy_scores:
            return _fallback_hook(0.0, target_duration, "No audio data")
        best = max(energy_scores, key=lambda x: x["energy_score"])
        print(f"[hook_detector] Audio-only hook: {best['start']:.1f}s - {best['end']:.1f}s")
        return {
            "start": best["start"],
            "end": best["end"],
            "score": best["energy_score"],
            "reason": "Highest audio energy segment (no transcript)",
            "method": "audio",
            "preview_text": "",
        }

    candidates = build_hook_candidates(segments, target_duration, audio_path, max_candidates=top_k)
    if not candidates:
        return _fallback_hook(segments[0]["start"], min(segments[0]["start"] + target_duration, segments[-1]["end"]), "No valid candidates")

    print(f"[hook_detector] Top heuristic scores: {[round(c['heuristic_score'], 2) for c in candidates[:3]]}")

    best = candidates[0]
    if use_ai:
        from src.ai_client import is_available, score_hook_candidates

        if is_available():
            ai_pick = score_hook_candidates(candidates, target_duration=target_duration)
            if ai_pick:
                chosen = next((c for c in candidates if c["id"] == ai_pick.get("best_id")), None)
                if chosen:
                    start, end = chosen["start"], chosen["end"]
                    preview = chosen["text"][:200]
                    base_score = chosen["heuristic_score"]
                elif "best_start" in ai_pick:
                    start, end = _snap_to_segments(
                        float(ai_pick["best_start"]),
                        float(ai_pick["best_end"]),
                        segments,
                        target_duration,
                    )
                    preview = ai_pick.get("preview_text", "")
                    base_score = best["heuristic_score"]
                else:
                    start, end = best["start"], best["end"]
                    preview = best["text"][:200]
                    base_score = best["heuristic_score"]

                score = float(ai_pick.get("score", base_score))
                if score >= 0.45:
                    print(f"[hook_detector] AI hook #{ai_pick.get('best_id')}: {start:.1f}s - {end:.1f}s (score={score:.2f})")
                    print(f"[hook_detector] Reason: {ai_pick.get('reason', '')}")
                    if preview:
                        print(f"[hook_detector] Preview: {preview[:120]}...")
                    return {
                        "start": start,
                        "end": end,
                        "score": score,
                        "reason": ai_pick.get("reason", "AI selected hook"),
                        "method": "ai",
                        "preview_text": preview,
                    }
                print(f"[hook_detector] AI score too low ({score:.2f}), using heuristic best")

    start, end = best["start"], best["end"]
    print(f"[hook_detector] Heuristic hook: {start:.1f}s - {end:.1f}s (score={best['heuristic_score']:.2f})")
    print(f"[hook_detector] Preview: {best['text'][:120]}...")
    return {
        "start": start,
        "end": end,
        "score": best["heuristic_score"],
        "reason": "Best heuristic score — strong opening, energy, and speech pace",
        "method": "heuristic",
        "preview_text": best["text"][:200],
    }


def _fallback_hook(start: float, end: float, reason: str) -> dict:
    return {
        "start": start,
        "end": end,
        "score": 0.0,
        "reason": reason,
        "method": "fallback",
        "preview_text": "",
    }


def extract_hook_clip(
    video_path: str,
    hook_result: dict,
    output_path: str,
    crf: int = 23,
) -> str:
    """Cut the hook segment using FFmpeg (fast)."""
    start = float(hook_result["start"])
    end = float(hook_result["end"])
    duration = max(0.5, end - start)

    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[hook_detector] Extracting hook clip {start:.1f}s - {end:.1f}s ({duration:.1f}s) ...")
    run_ffmpeg(
        [
            "-ss", f"{start:.3f}",
            "-i", str(Path(video_path).resolve()),
            "-t", f"{duration:.3f}",
            *video_encoder_args(crf=crf, fast=True),
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ],
        label="hook_detector/extract",
    )
    print(f"[hook_detector] Hook clip saved: {output_path}")
    return output_path
