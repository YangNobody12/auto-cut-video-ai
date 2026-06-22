"""
ai_client.py
Multi-provider AI router for hook scoring.
Supports OpenAI (GPT-4o), Anthropic (Claude), Google Gemini, OpenTyphoon (Typhoon).
Falls back to None if no API key is configured.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PROVIDER = os.getenv("AI_PROVIDER", "").strip().lower()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
TYPHOON_KEY = (
    os.getenv("THPHOON_API_KEY", "").strip()
    or os.getenv("TYPHOON_API_KEY", "").strip()
    or os.getenv("OPENTYPHOON_API_KEY", "").strip()
)
TYPHOON_BASE_URL = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1").strip()
TYPHOON_MODEL = os.getenv("TYPHOON_MODEL", "typhoon-v2.5-30b-a3b-instruct").strip()
TYPHOON_TEMPERATURE = float(os.getenv("TYPHOON_TEMPERATURE", "0.6"))
TYPHOON_TOP_P = float(os.getenv("TYPHOON_TOP_P", "0.6"))
TYPHOON_MAX_TOKENS = int(os.getenv("TYPHOON_MAX_TOKENS", "512"))

_TYPHOON_SYSTEM_PROMPT = (
    "You are an AI assistant named Typhoon created by SCB 10X to be helpful, harmless, and honest. "
    "Typhoon is happy to help with analysis, question answering, math, coding, creative writing, "
    "teaching, role-play, general discussion, and all sorts of other tasks. Typhoon responds directly "
    "to all human messages without unnecessary affirmations or filler phrases like \"Certainly!\", "
    "\"Of course!\", \"Absolutely!\", \"Great!\", \"Sure!\", etc. Specifically, Typhoon avoids starting "
    "responses with the word \"Certainly\" in any way. Typhoon follows this information in all languages, "
    "and always responds to the user in the language they use or request. Typhoon is now being connected "
    "with a human. Write in fluid, conversational prose, Show genuine interest in understanding requests, "
    "Express appropriate emotions and empathy."
)

_HOOK_TASK_PROMPT = """You are a viral short-form video editor (TikTok / Reels / YouTube Shorts) specializing in Thai content.

You will receive numbered clip CANDIDATES with timestamps and transcript text.
Pick the ONE candidate that works best as a standalone hook clip.

REJECT candidates that:
- Start with greetings or filler (สวัสดี, วันนี้, โอเค, เอ่อ, hello, ok, welcome)
- Need prior context — viewer must understand without watching earlier parts
- Slow setup with no punch in the first 3 seconds
- Are mid-sentence cuts or incoherent

PREFER candidates that:
- Open with a bold claim, provocative question, or surprising fact
- Create curiosity ("รู้ไหมว่า...", "อย่าเพิ่ง...", "นี่คือเหตุผลที่...")
- Have high energy, emotion, controversy, or a clear payoff tease
- Sound cool and shareable — something you'd stop scrolling for

Return JSON only:
{
  "best_id": <candidate number>,
  "score": <0.0-1.0 how viral/engaging this hook is>,
  "reason": "<1-2 sentences in Thai explaining why this hook works>"
}"""

_HOOK_SYSTEM_PROMPT = """You are a professional video editor and content strategist.Your task is to analyze a video transcript and identify the BEST hook segment.

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


def _build_hook_candidates_prompt(candidates: list[dict], target_duration: float) -> str:
    lines = [
        f"Target clip length: ~{target_duration:.0f} seconds",
        f"Number of candidates: {len(candidates)}",
        "",
    ]
    for c in candidates:
        preview = c.get("text", "").replace("\n", " ")
        if len(preview) > 350:
            preview = preview[:350] + "..."
        lines.append(
            f"[{c['id']}] {c['start']:.1f}s - {c['end']:.1f}s "
            f"(heuristic={c.get('heuristic_score', 0):.2f})"
        )
        lines.append(f"    \"{preview}\"")
        lines.append("")
    lines.append("Pick the best_id for the most engaging standalone hook. Return JSON only.")
    return "\n".join(lines)


def _build_hook_prompt(segments: list[dict], target_duration: float) -> str:
    lines = [
        f"Video transcript segments (each with start/end timestamps in seconds):\n",
    ]
    for seg in segments:
        lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
    lines.append(f"\nTarget hook duration: ~{target_duration:.0f} seconds")
    lines.append("\nIdentify the single best hook segment and return JSON.")
    return "\n".join(lines)


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object in model response")
    return json.loads(text[start:end])


def _typhoon_client():
    from openai import OpenAI

    return OpenAI(api_key=TYPHOON_KEY, base_url=TYPHOON_BASE_URL)


def _typhoon_completion(messages: list[dict]) -> str:
    client = _typhoon_client()
    kwargs = {
        "model": TYPHOON_MODEL,
        "messages": messages,
        "temperature": TYPHOON_TEMPERATURE,
        "top_p": TYPHOON_TOP_P,
        "frequency_penalty": 0,
        "stream": False,
    }
    try:
        response = client.chat.completions.create(
            **kwargs,
            max_completion_tokens=TYPHOON_MAX_TOKENS,
        )
    except TypeError:
        response = client.chat.completions.create(
            **kwargs,
            max_tokens=TYPHOON_MAX_TOKENS,
        )
    return response.choices[0].message.content or ""


def score_hook_candidates(candidates: list[dict], target_duration: float = 30.0) -> dict | None:
    """Rank pre-built hook candidates with AI."""
    if not candidates:
        return None

    provider = PROVIDER or ("typhoon" if TYPHOON_KEY else "openai" if OPENAI_KEY else "")
    print(f"[ai_client] Ranking {len(candidates)} hook candidates with provider: {provider or 'none'}")

    prompt = _build_hook_candidates_prompt(candidates, target_duration)

    if provider in ("typhoon", "opentyphoon") and TYPHOON_KEY:
        return _score_hook_candidates_typhoon(prompt)
    if provider == "openai" and OPENAI_KEY:
        return _score_hook_candidates_openai(prompt)
    if provider == "anthropic" and ANTHROPIC_KEY:
        return _score_hook_candidates_anthropic(prompt)
    if provider == "gemini" and GOOGLE_KEY:
        return _score_hook_candidates_gemini(prompt)
    if TYPHOON_KEY:
        return _score_hook_candidates_typhoon(prompt)
    if OPENAI_KEY:
        return _score_hook_candidates_openai(prompt)
    return None


def _score_hook_candidates_typhoon(prompt: str) -> dict | None:
    try:
        messages = [
            {"role": "system", "content": _TYPHOON_SYSTEM_PROMPT},
            {"role": "user", "content": f"{_HOOK_TASK_PROMPT}\n\n{prompt}"},
        ]
        hook_temp = min(TYPHOON_TEMPERATURE, 0.4)
        client = _typhoon_client()
        try:
            response = client.chat.completions.create(
                model=TYPHOON_MODEL,
                messages=messages,
                temperature=hook_temp,
                top_p=TYPHOON_TOP_P,
                frequency_penalty=0,
                response_format={"type": "json_object"},
                max_completion_tokens=max(TYPHOON_MAX_TOKENS, 1024),
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            text = _typhoon_completion(messages)
            return _parse_json_response(text)
    except Exception as e:
        print(f"[ai_client] Typhoon hook ranking error: {e}")
        return None


def _score_hook_candidates_openai(prompt: str) -> dict | None:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _HOOK_SYSTEM_PROMPT},
                {"role": "user", "content": f"{_HOOK_TASK_PROMPT}\n\n{prompt}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[ai_client] OpenAI hook ranking error: {e}")
        return None


def _score_hook_candidates_anthropic(prompt: str) -> dict | None:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            system=_HOOK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{_HOOK_TASK_PROMPT}\n\n{prompt}"}],
            temperature=0.3,
        )
        return _parse_json_response(message.content[0].text.strip())
    except Exception as e:
        print(f"[ai_client] Anthropic hook ranking error: {e}")
        return None


def _score_hook_candidates_gemini(prompt: str) -> dict | None:
    try:
        import google.generativeai as genai

        genai.configure(api_key=GOOGLE_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=_HOOK_SYSTEM_PROMPT,
        )
        response = model.generate_content(
            f"{_HOOK_TASK_PROMPT}\n\n{prompt}",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"[ai_client] Gemini hook ranking error: {e}")
        return None


def score_hook_typhoon(segments: list[dict], target_duration: float = 30.0) -> dict | None:
    try:
        messages = [
            {"role": "system", "content": _TYPHOON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{_HOOK_TASK_PROMPT}\n\n{_build_hook_prompt(segments, target_duration)}",
            },
        ]
        try:
            client = _typhoon_client()
            response = client.chat.completions.create(
                model=TYPHOON_MODEL,
                messages=messages,
                temperature=TYPHOON_TEMPERATURE,
                top_p=TYPHOON_TOP_P,
                frequency_penalty=0,
                response_format={"type": "json_object"},
                max_completion_tokens=TYPHOON_MAX_TOKENS,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            text = _typhoon_completion(messages)
            return _parse_json_response(text)
    except Exception as e:
        print(f"[ai_client] Typhoon error: {e}")
        return None


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
        return _parse_json_response(text)
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
    """Legacy: build candidates internally then rank with AI."""
    from src.hook_detector import build_hook_candidates

    candidates = build_hook_candidates(segments, target_duration, max_candidates=8)
    if not candidates:
        return None
    return score_hook_candidates(candidates, target_duration)


def is_available() -> bool:
    """Return True if at least one AI provider is configured."""
    return bool(TYPHOON_KEY or OPENAI_KEY or ANTHROPIC_KEY or GOOGLE_KEY)
