"""
clip_exporter.py
Handles video resizing, cropping, and final export via FFmpeg (fast).
"""
from pathlib import Path
from typing import Literal

from src.ffmpeg_utils import copy_or_reencode, probe_video, run_ffmpeg, video_encoder_args


ClipMode = Literal["short", "long"]

SIZE_PRESETS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "4:3": (1440, 1080),
    "21:9": (2560, 1080),
}


def parse_size(size_str: str) -> tuple[int, int]:
    size_str = size_str.strip()
    if size_str in SIZE_PRESETS:
        return SIZE_PRESETS[size_str]
    if "x" in size_str.lower():
        parts = size_str.lower().split("x")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    raise ValueError(
        f"Unknown size '{size_str}'. Use presets ({', '.join(SIZE_PRESETS.keys())}) or WxH (e.g. 1280x720)."
    )


def export_clip(
    video_path: str,
    output_path: str,
    mode: ClipMode = "long",
    size: str = "16:9",
    duration: float | None = None,
    start_time: float = 0.0,
    fps: int = 30,
    crf: int = 23,
    audio_bitrate: str = "192k",
) -> str:
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    target_w, target_h = parse_size(size)
    info = probe_video(video_path)
    src_w, src_h = info["width"], info["height"]

    end_time = info["duration"]
    if mode == "short":
        clip_duration = duration if duration is not None else min(60.0, info["duration"])
        end_time = min(start_time + clip_duration, info["duration"])

    needs_reencode = (
        mode == "short"
        or src_w != target_w
        or src_h != target_h
        or abs(info["fps"] - fps) > 0.5
    )

    if not needs_reencode and Path(video_path) != Path(output_path):
        print(f"[clip_exporter] Copying to output (no resize needed) ...")
        return copy_or_reencode(video_path, output_path)

    print(f"[clip_exporter] Exporting [{mode}] -> {target_w}x{target_h} from {Path(video_path).name}")

    vf_parts: list[str] = []
    if mode == "short":
        vf_parts.append(f"trim=start={start_time:.3f}:end={end_time:.3f},setpts=PTS-STARTPTS")

    if (src_w, src_h) != (target_w, target_h):
        target_ratio = target_w / target_h
        src_ratio = src_w / src_h if src_h else target_ratio
        if src_ratio > target_ratio:
            scaled_h = target_h
            scaled_w = int(src_w * target_h / src_h)
        else:
            scaled_w = target_w
            scaled_h = int(src_h * target_w / src_w)
        x1 = max(0, (scaled_w - target_w) // 2)
        y1 = max(0, (scaled_h - target_h) // 2)
        vf_parts.append(f"scale={scaled_w}:{scaled_h}")
        vf_parts.append(f"crop={target_w}:{target_h}:{x1}:{y1}")

    args = ["-i", video_path]
    if vf_parts:
        args.extend(["-vf", ",".join(vf_parts)])
    if mode == "short":
        args.extend(["-af", f"atrim=start={start_time:.3f}:end={end_time:.3f},asetpts=PTS-STARTPTS"])

    args.extend([
        *video_encoder_args(crf=crf),
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-r", str(fps),
        output_path,
    ])
    run_ffmpeg(args, label="clip_exporter")
    print(f"[clip_exporter] Export complete: {output_path}")
    return output_path


def get_video_info(video_path: str) -> dict:
    info = probe_video(video_path)
    return {
        "path": str(Path(video_path).resolve()),
        "duration": round(info["duration"], 2),
        "size": [info["width"], info["height"]],
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "aspect_ratio": f"{info['width']}:{info['height']}",
        "has_audio": True,
    }


def list_size_presets() -> dict:
    return {name: f"{w}x{h}" for name, (w, h) in SIZE_PRESETS.items()}
