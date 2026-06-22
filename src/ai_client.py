"""
ai_client.py
Multi-provider AI router for hook scoring.
Supports OpenAI (GPT-4o), Anthropic (Claude), Google Gemini.
Falls back to None if no API key is configured.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PROVIDER = os.getenv("AI_PROVIDER", "").strip().lower()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

_HOOK_SYSTEM_PROMPT = """You are a professional video editor and content strategist.
Your task is to analyze a video transcript and identify the BEST hook segment.

A good hook should:
- Be attention-grabbing in the first 3 seconds
- Create curiosity or emotional impact
- Work well as the opening of a short-form video (TikTok, Reels, YouTube Shorts)
- Contain a strong statement, question, or surprising fact

Return JSON only in this exact format:
{
  "best_start": <seconds as float>,
  "best_end": <seconds as float>,
  "score": <0.0-1.0>,
  "reason": "<short explanation in 1-2 sentences>"
}"""


def _build_hook_prompt(segments: list[dict], target_duration: float) -> str:
    lines = [
        f"Video transcript segments (each with start/end timestamps in seconds):\n",
    ]
    for seg in segments:
        lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
    lines.append(f"\nTarget hook duration: ~{target_duration:.0f} seconds")
    lines.append("\nIdentify the single best hook segment and return JSON.")
    return "\n".join(lines)


def score_hook_openai(segments: list[dict], target_duration: float = 30.0) -> dict | None:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _HOOK_SYSTEM_PROMPT},
                {"role": "user", "content": _build_hook_prompt(segments, target_duration)},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[ai_client] OpenAI error: {e}")
        return None


def score_hook_anthropic(segments: list[dict], target_duration: float = 30.0) -> dict | None:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        prompt = _build_hook_prompt(segments, target_duration)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            system=_HOOK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        import json
        text = message.content[0].text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception as e:
        print(f"[ai_client] Anthropic error: {e}")
        return None


def score_hook_gemini(segments: list[dict], target_duration: float = 30.0) -> dict | None:
    try:
        import google.generativeai as genai
        import json
        genai.configure(api_key=GOOGLE_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=_HOOK_SYSTEM_PROMPT,
        )
        prompt = _build_hook_prompt(segments, target_duration)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"[ai_client] Gemini error: {e}")
        return None


def score_hook(segments: list[dict], target_duration: float = 30.0) -> dict | None:
    """
    Route hook scoring to the configured AI provider.

    Args:
        segments: List of {"start": float, "end": float, "text": str}
        target_duration: Desired hook clip length in seconds.

    Returns:
        Dict with best_start, best_end, score, reason — or None if unavailable.
    """
    if not segments:
        return None

    provider = PROVIDER
    if not provider:
        if OPENAI_KEY:
            provider = "openai"
        elif ANTHROPIC_KEY:
            provider = "anthropic"
        elif GOOGLE_KEY:
            provider = "gemini"
        else:
            print("[ai_client] No AI provider configured. Using audio-only analysis.")
            return None

    print(f"[ai_client] Scoring hook with provider: {provider}")

    if provider == "openai":
        return score_hook_openai(segments, target_duration)
    elif provider == "anthropic":
        return score_hook_anthropic(segments, target_duration)
    elif provider == "gemini":
        return score_hook_gemini(segments, target_duration)
    else:
        print(f"[ai_client] Unknown provider '{provider}'. Skipping AI scoring.")
        return None


def is_available() -> bool:
    """Return True if at least one AI provider is configured."""
    return bool(OPENAI_KEY or ANTHROPIC_KEY or GOOGLE_KEY)
