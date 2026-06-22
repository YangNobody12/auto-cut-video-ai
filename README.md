# Auto Cut AI

Python-based automatic video editor with AI-powered hook detection, silence removal, styled subtitles, background music, and flexible export sizes.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) — must be on PATH

### 2. Setup

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 3. Configure AI (optional)

Copy `.env.example` to `.env` and add your key:

```ini
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Supported providers: `typhoon` (OpenTyphoon), `openai` (GPT-4o), `anthropic` (Claude), `gemini` (Gemini 1.5 Pro)

**OpenTyphoon (แนะนำสำหรับภาษาไทย):**
```ini
AI_PROVIDER=typhoon
THPHOON_API_KEY=sk-...
# optional: TYPHOON_MODEL=typhoon-v2.5-30b-a3b-instruct
```
Uses the [OpenTyphoon API](https://docs.opentyphoon.ai/en/) (OpenAI-compatible).

### 4. Noise reduction (optional)

`--denoise` uses [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) via the `deep-filter` CLI.
On first use, the Windows binary is auto-downloaded to `tools/deep-filter/`.
You can also install manually from the [releases page](https://github.com/Rikorose/DeepFilterNet/releases).

---

## Usage

```powershell
# Check environment
python main.py --check-env

# Get video info
python main.py --info --input video.mp4

# List available size presets
python main.py --list-sizes
```

### Examples

```powershell
# Remove silence and add subtitles (full video, YouTube format)
python main.py --input video.mp4 --remove-silence --subtitle

# Denoise + silence removal + subtitles
python main.py --input video.mp4 --denoise --remove-silence --subtitle --sub-language th

# AI hook detection → 30s TikTok clip with karaoke subtitles
python main.py --input video.mp4 --hook --mode short --duration 30 --size 9:16 --subtitle --sub-style karaoke

# Full edited video with background music + subtitle
python main.py --input video.mp4 --remove-silence --subtitle --music assets/music/bgm.mp3 --music-volume 0.2 --size 16:9

# Transcribe only (no video editing)
python main.py --input video.mp4 --transcribe-only --whisper-model medium

# Batch process a folder
python main.py --batch input/ --remove-silence --subtitle --size 9:16
```

---

## Features

| Feature | Flag | Description |
|---|---|---|
| Noise reduction | `--denoise` | DeepFilterNet background noise removal |
| Silence removal | `--remove-silence` | Cuts dead-air segments |
| Hook detection | `--hook` | Finds best engagement moment |
| AI hook scoring | `--hook-ai` | Uses GPT-4o/Claude/Gemini to analyze transcript |
| Subtitles | `--subtitle` | Whisper transcription + burn-in |
| Subtitle styles | `--sub-style` | default, karaoke, box, gradient, minimal |
| Background music | `--music FILE` | Add BGM with auto-ducking |
| Screen size | `--size PRESET` | 9:16, 16:9, 1:1, 4:5 or custom WxH |
| Short clip | `--mode short` | Extract N-second highlight |
| Batch mode | `--batch DIR` | Process all videos in a directory |

---

## Subtitle Styles

| Style | Description |
|---|---|
| `default` | White text, black outline |
| `karaoke` | Yellow text, word-by-word style |
| `box` | Text with semi-transparent background box |
| `gradient` | Gold text with orange stroke |
| `minimal` | Small, clean text near bottom |

---

## Screen Size Presets

| Preset | Resolution | Platform |
|---|---|---|
| `9:16` | 1080×1920 | TikTok, Reels, YouTube Shorts |
| `16:9` | 1920×1080 | YouTube, standard |
| `1:1` | 1080×1080 | Instagram square |
| `4:5` | 1080×1350 | Instagram portrait |
| `4:3` | 1440×1080 | Traditional |
| Custom | `WxH` | e.g. `1280x720` |

---

## Project Structure

```
auto-cut-ai/
├── main.py              ← CLI entry point
├── config.yaml          ← Default settings
├── requirements.txt
├── .env                 ← API keys (create from .env.example)
├── .env.example
├── input/               ← Drop source videos here
├── output/              ← Exported videos saved here
├── temp/                ← Working files (auto-cleaned)
├── tools/
│   └── deep-filter/     ← deep-filter binary (auto-downloaded on first --denoise)
├── assets/
│   ├── music/           ← Background music files
│   └── fonts/           ← Custom fonts
└── src/
    ├── editor.py        ← Pipeline orchestrator
    ├── audio_extractor.py
    ├── noise_reducer.py   ← DeepFilterNet noise removal
    ├── transcriber.py
    ├── silence_remover.py
    ├── hook_detector.py
    ├── ai_client.py
    ├── subtitle_renderer.py
    ├── music_mixer.py
    └── clip_exporter.py
```
