"""
music_mixer.py
Mixes background music into a video with:
- Volume control
- Audio ducking (lower BGM when speech is detected)
- Fade in / fade out
- Auto-loop (repeats BGM to match video duration)
"""
from pathlib import Path


def mix_music(
    video_path: str,
    music_path: str,
    output_path: str,
    music_volume: float = 0.15,
    duck_volume: float = 0.07,
    fade_in_sec: float = 1.0,
    fade_out_sec: float = 2.0,
    loop: bool = True,
    ducking: bool = True,
    speech_segments: list[dict] | None = None,
) -> str:
    """
    Add background music to a video.

    Args:
        video_path: Source video file path.
        music_path: Background music file path (MP3/WAV/etc.).
        output_path: Output video file path.
        music_volume: BGM volume ratio (0.0 - 1.0). Default 0.15 (15%).
        duck_volume: BGM volume during speech if ducking is on. Default 0.07 (7%).
        fade_in_sec: BGM fade-in duration in seconds.
        fade_out_sec: BGM fade-out duration in seconds.
        loop: Repeat BGM if shorter than video.
        ducking: Reduce BGM volume during speech segments.
        speech_segments: Transcript segments for ducking [{start, end, text}].

    Returns:
        Path to output video with mixed audio.
    """
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
    import numpy as np

    print(f"[music_mixer] Loading video: {Path(video_path).name}")
    video = VideoFileClip(video_path)
    video_duration = video.duration

    print(f"[music_mixer] Loading music: {Path(music_path).name}")
    music = AudioFileClip(music_path)

    if loop and music.duration < video_duration:
        repeats = int(video_duration / music.duration) + 2
        music_parts = [music] * repeats
        music = concatenate_audioclips(music_parts)
        print(f"[music_mixer] Looped BGM x{repeats} to cover {video_duration:.1f}s")

    music = music.subclipped(0, video_duration)

    if fade_in_sec > 0:
        from moviepy.audio.fx import AudioFadeIn
        music = music.with_effects([AudioFadeIn(fade_in_sec)])
    if fade_out_sec > 0:
        from moviepy.audio.fx import AudioFadeOut
        music = music.with_effects([AudioFadeOut(fade_out_sec)])

    if ducking and speech_segments:
        music = _apply_ducking(music, speech_segments, music_volume, duck_volume, video_duration)
    else:
        from moviepy.audio.fx import MultiplyVolume
        music = music.with_effects([MultiplyVolume(music_volume)])

    original_audio = video.audio
    if original_audio is not None:
        mixed_audio = CompositeAudioClip([original_audio, music])
    else:
        mixed_audio = music

    final = video.set_audio(mixed_audio)
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[music_mixer] Writing mixed video to {output_path} ...")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    final.close()
    video.close()
    music.close()
    print(f"[music_mixer] Done.")
    return output_path


def _apply_ducking(
    music_clip,
    speech_segments: list[dict],
    full_volume: float,
    duck_volume: float,
    total_duration: float,
):
    """
    Apply volume ducking: lower BGM volume during speech, restore during silence.
    Uses piecewise constant volume changes via moviepy's volumex with time intervals.
    """
    from moviepy.editor import AudioClip
    import numpy as np

    FADE_SEC = 0.3

    events: list[tuple[float, float]] = []
    for seg in speech_segments:
        events.append((max(0, seg["start"] - FADE_SEC), "duck_start"))
        events.append((min(total_duration, seg["end"] + FADE_SEC), "duck_end"))

    def vol_at(t):
        if isinstance(t, np.ndarray):
            return np.array([_point_volume(ti, speech_segments, full_volume, duck_volume, FADE_SEC) for ti in t])
        return _point_volume(t, speech_segments, full_volume, duck_volume, FADE_SEC)

    class DuckedAudio:
        def __init__(self, clip):
            self._clip = clip

        def make_frame(self, t):
            frame = self._clip.make_frame(t)
            v = vol_at(t) if not isinstance(t, np.ndarray) else vol_at(t)
            if isinstance(v, np.ndarray):
                return (frame.T * v).T
            return frame * v

    try:
        from moviepy.audio.AudioClip import AudioArrayClip
        from moviepy.audio.fx import MultiplyVolume
        import numpy as np

        sr = 44100
        n_samples = int(total_duration * sr)
        t_arr = np.linspace(0, total_duration, n_samples, endpoint=False)
        raw = music_clip.get_frame(t_arr).T if hasattr(music_clip, 'get_frame') else music_clip.make_frame(t_arr).T
        vol_arr = _point_volume_array(t_arr, speech_segments, full_volume, duck_volume, FADE_SEC)
        ducked = raw * vol_arr
        return AudioArrayClip(ducked.T, fps=sr).subclipped(0, total_duration)
    except Exception as e:
        print(f"[music_mixer] Ducking fallback: {e}")
        from moviepy.audio.fx import MultiplyVolume
        return music_clip.with_effects([MultiplyVolume(full_volume)])


def _point_volume(t: float, segments: list[dict], full_vol: float, duck_vol: float, fade: float) -> float:
    for seg in segments:
        s, e = seg["start"], seg["end"]
        if s - fade <= t <= s:
            ratio = (t - (s - fade)) / fade
            return full_vol * (1 - ratio) + duck_vol * ratio
        elif s < t < e:
            return duck_vol
        elif e <= t <= e + fade:
            ratio = (t - e) / fade
            return duck_vol * (1 - ratio) + full_vol * ratio
    return full_vol


def _point_volume_array(t_arr, segments, full_vol, duck_vol, fade):
    import numpy as np
    vol = np.full(len(t_arr), full_vol, dtype=np.float32)
    for seg in segments:
        s, e = seg["start"], seg["end"]
        fade_in_mask = (t_arr >= s - fade) & (t_arr < s)
        ratio = (t_arr[fade_in_mask] - (s - fade)) / fade
        vol[fade_in_mask] = full_vol * (1 - ratio) + duck_vol * ratio

        speech_mask = (t_arr >= s) & (t_arr < e)
        vol[speech_mask] = duck_vol

        fade_out_mask = (t_arr >= e) & (t_arr <= e + fade)
        ratio = (t_arr[fade_out_mask] - e) / fade
        vol[fade_out_mask] = duck_vol * (1 - ratio) + full_vol * ratio

    return vol.reshape(-1, 1)


def list_music_files(music_dir: str = "assets/music") -> list[str]:
    """Return list of music files in the assets/music directory."""
    music_dir = Path(music_dir)
    if not music_dir.exists():
        return []
    extensions = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
    return [str(f) for f in music_dir.iterdir() if f.suffix.lower() in extensions]
