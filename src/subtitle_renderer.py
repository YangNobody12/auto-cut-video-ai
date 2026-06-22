"""
subtitle_renderer.py
Burns styled subtitles onto video using FFmpeg (fast) with MoviePy fallback.
"""
import re
import sys
from pathlib import Path
from typing import Literal

from src.ffmpeg_utils import escape_filter_path, probe_video, run_ffmpeg, video_encoder_args


SubtitleStyle = Literal["default", "karaoke", "box", "gradient", "minimal"]

STYLE_PRESETS = {
    "default": {
        "fontsize": 40,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 2,
        "font": "Arial-Bold",
        "alignment": 2,
        "margin_v": 40,
    },
    "karaoke": {
        "fontsize": 44,
        "color": "yellow",
        "stroke_color": "black",
        "stroke_width": 2,
        "font": "Arial-Bold",
        "alignment": 2,
        "margin_v": 40,
    },
    "box": {
        "fontsize": 38,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 0,
        "font": "Arial",
        "alignment": 2,
        "margin_v": 40,
        "border_style": 3,
        "back_colour": "&H80000000",
    },
    "gradient": {
        "fontsize": 42,
        "color": "#FFD700",
        "stroke_color": "#FF4500",
        "stroke_width": 3,
        "font": "Arial-Bold",
        "alignment": 2,
        "margin_v": 40,
    },
    "minimal": {
        "fontsize": 30,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 1,
        "font": "Arial",
        "alignment": 2,
        "margin_v": 24,
    },
}


def _resolve_font(font_name: str, font_path: str | None = None) -> str:
    if font_path and Path(font_path).exists():
        return Path(font_path).stem

    if sys.platform == "win32":
        named = {
            "Arial-Bold": ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"],
            "Arial": ["C:/Windows/Fonts/arial.ttf"],
        }
        for path in named.get(font_name, []):
            if Path(path).exists():
                return Path(path).stem
        for path in [
            "C:/Windows/Fonts/LeelawUI.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]:
            if Path(path).exists():
                return Path(path).stem

    assets_font = Path("assets/fonts")
    if assets_font.is_dir():
        for ttf in assets_font.glob("*.ttf"):
            return ttf.stem

    return "Arial"


def _seconds_to_srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _seconds_to_ass_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    cs = int(round((s - int(s)) * 100))
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def segments_to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_seconds_to_srt_time(seg['start'])} --> {_seconds_to_srt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def segments_to_ass(
    segments: list[dict],
    style: SubtitleStyle = "default",
    font_path: str | None = None,
    video_w: int = 1280,
    video_h: int = 720,
) -> str:
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["default"])
    font_name = _resolve_font(preset.get("font", "Arial"), font_path)
    fontsize = preset["fontsize"]
    outline = preset.get("stroke_width", 2)
    border_style = preset.get("border_style", 1)
    back_colour = preset.get("back_colour", "&H00000000")
    alignment = preset.get("alignment", 2)
    margin_v = preset.get("margin_v", 40)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,{back_colour},-1,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for seg in segments:
        text = seg["text"].strip().replace("\n", "\\N")
        if not text:
            continue
        events.append(
            f"Dialogue: 0,{_seconds_to_ass_time(seg['start'])},{_seconds_to_ass_time(seg['end'])},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


def _render_subtitles_ffmpeg(
    video_path: str,
    segments: list[dict],
    output_path: str,
    style: SubtitleStyle = "default",
    font_path: str | None = None,
    crf: int = 23,
) -> str:
    info = probe_video(video_path)
    video_path = str(Path(video_path).resolve())
    output_path = str(Path(output_path).resolve())
    work_dir = Path(output_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    ass_name = "_burn_subtitles.ass"
    ass_path = work_dir / ass_name
    ass_path.write_text(
        segments_to_ass(
            segments,
            style=style,
            font_path=font_path,
            video_w=info["width"] or 1280,
            video_h=info["height"] or 720,
        ),
        encoding="utf-8-sig",
    )

    args = [
        "-i", video_path,
        "-vf", f"ass={ass_name}",
        *video_encoder_args(crf=crf),
        "-c:a", "copy",
        Path(output_path).name,
    ]
    print(f"[subtitle_renderer] Burning {len(segments)} subtitles via FFmpeg ...")
    run_ffmpeg(args, label="subtitle_renderer", cwd=work_dir)
    ass_path.unlink(missing_ok=True)
    print(f"[subtitle_renderer] Done: {output_path}")
    return output_path


def _render_subtitles_moviepy(
    video_path: str,
    segments: list[dict],
    output_path: str,
    style: SubtitleStyle = "default",
    font_path: str | None = None,
    fontsize: int | None = None,
    color: str | None = None,
    position: tuple | None = None,
    max_chars_per_line: int = 40,
) -> str:
    from moviepy import CompositeVideoClip, TextClip, VideoFileClip

    preset = {**STYLE_PRESETS.get(style, STYLE_PRESETS["default"])}
    resolved_font = _resolve_font(preset.get("font", "Arial"), font_path)
    if fontsize:
        preset["fontsize"] = fontsize
    if color:
        preset["color"] = color

    video = VideoFileClip(video_path)
    video_w, video_h = video.size

    def _wrap_text(text: str, max_chars: int) -> str:
        words = text.split()
        lines, current = [], ""
        for word in words:
            if len(current) + len(word) + 1 > max_chars and current:
                lines.append(current.strip())
                current = word + " "
            else:
                current += word + " "
        if current.strip():
            lines.append(current.strip())
        return "\n".join(lines)

    y_pos = int(video_h * 0.85)
    subtitle_clips = []
    for seg in segments:
        start = seg["start"]
        end = min(seg["end"], video.duration)
        if start >= video.duration:
            continue
        try:
            txt_clip = (
                TextClip(
                    text=_wrap_text(seg["text"], max_chars_per_line),
                    font_size=preset["fontsize"],
                    color=preset.get("color", "white"),
                    font=resolved_font,
                    method="caption",
                    size=(int(video_w * 0.9), None),
                    stroke_color=preset.get("stroke_color"),
                    stroke_width=preset.get("stroke_width", 0),
                )
                .with_start(start)
                .with_duration(max(0.1, end - start))
                .with_position(("center", y_pos), relative=True)
            )
            subtitle_clips.append(txt_clip)
        except Exception as exc:
            print(f"[subtitle_renderer] Warning: segment at {start:.1f}s skipped: {exc}")

    if not subtitle_clips:
        video.close()
        return video_path

    final = CompositeVideoClip([video, *subtitle_clips])
    output_path = str(Path(output_path).resolve())
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="veryfast")
    final.close()
    video.close()
    return output_path


def render_subtitles(
    video_path: str,
    segments: list[dict],
    output_path: str,
    style: SubtitleStyle = "default",
    font_path: str | None = None,
    fontsize: int | None = None,
    color: str | None = None,
    position: tuple | None = None,
    max_chars_per_line: int = 40,
    method: str = "ffmpeg",
    crf: int = 23,
) -> str:
    print(f"[subtitle_renderer] Rendering subtitles with style='{style}' on {Path(video_path).name} ...")
    if method == "moviepy":
        return _render_subtitles_moviepy(
            video_path, segments, output_path, style, font_path, fontsize, color, position, max_chars_per_line
        )
    try:
        return _render_subtitles_ffmpeg(video_path, segments, output_path, style, font_path, crf=crf)
    except Exception as exc:
        print(f"[subtitle_renderer] FFmpeg burn failed ({exc})")
        raise


def burn_srt_file(
    video_path: str,
    srt_path: str,
    output_path: str,
    style: SubtitleStyle = "default",
) -> str:
    segments = _parse_srt(srt_path)
    return render_subtitles(video_path, segments, output_path, style=style)


def _parse_srt(srt_path: str) -> list[dict]:
    text = Path(srt_path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"\d+\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n([\s\S]*?)(?=\n\n|\Z)",
        re.MULTILINE,
    )
    segments = []
    for m in pattern.finditer(text):
        start = _srt_time_to_sec(m.group(1))
        end = _srt_time_to_sec(m.group(2))
        txt = m.group(3).strip().replace("\n", " ")
        segments.append({"start": start, "end": end, "text": txt})
    return segments


def _srt_time_to_sec(t: str) -> float:
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
