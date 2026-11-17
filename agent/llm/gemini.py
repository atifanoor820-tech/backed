"""Gemini client (native generateContent REST API): JSON planning, text replies, screenshot grounding."""

import base64
import json
import logging

import httpx

from agent.config import get_settings

log = logging.getLogger("abbas.gemini")


class GeminiError(RuntimeError):
    pass


def _error_message(r: httpx.Response) -> str:
    try:
        return r.json()["error"]["message"][:300]
    except Exception:
        return r.text[:300]


RETRYABLE = {429, 500, 503, 504}


def generate(system: str, parts: list[dict], json_mode: bool = True, transport: httpx.BaseTransport | None = None,
             fast: bool = False, timeout_s: float | None = None, max_output_tokens: int | None = None) -> str:
    s = get_settings()
    if not s.gemini_enabled:
        raise GeminiError("Gemini is not configured (set ABBAS_GEMINI_API_KEY)")
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.2, **({"responseMimeType": "application/json"} if json_mode else {}),
                             **({"maxOutputTokens": max_output_tokens} if max_output_tokens else {})},
    }
    first = s.gemini_fast_model if fast else s.gemini_model
    models = list(dict.fromkeys([first, *(m.strip() for m in s.gemini_fallback_models.split(",") if m.strip())]))
    problems, r = [], None
    timeout = timeout_s or s.gemini_timeout_s
    with httpx.Client(timeout=timeout, transport=transport) as c:
        for model in models:
            url = f"{s.gemini_base_url.rstrip('/')}/models/{model}:generateContent"
            try:
                r = c.post(url, json=body, headers={"x-goog-api-key": s.gemini_api_key})
            except httpx.TimeoutException:
                problems.append(f"{model}: no reply in {timeout:.0f}s")
                r = None
                continue
            if r.status_code not in RETRYABLE:
                break
            problems.append(f"{model}: HTTP {r.status_code} {_error_message(r)[:80]}")
            log.warning("Gemini %s unavailable (%s); trying next free model", model, r.status_code)
    if r is None or r.status_code in RETRYABLE:
        quota = any("HTTP 429" in p for p in problems)
        raise GeminiError(("Gemini free-tier quota exhausted (resets midnight Pacific). " if quota else
                           "All free Gemini models are busy. ") + "; ".join(problems))
    if r.status_code != 200:
        raise GeminiError(f"Gemini HTTP {r.status_code}: {_error_message(r)}")
    data = r.json()
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError) as e:
        reason = data.get("promptFeedback") or data.get("candidates", [{}])[0].get("finishReason")
        raise GeminiError(f"Gemini returned no content ({reason})") from e


def generate_json(system: str, user: str, transport: httpx.BaseTransport | None = None, fast: bool = False,
                  timeout_s: float | None = None, max_output_tokens: int | None = None) -> dict:
    text = generate(system, [{"text": user}], json_mode=True, transport=transport, fast=fast,
                    timeout_s=timeout_s, max_output_tokens=max_output_tokens)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise GeminiError(f"Gemini did not return valid JSON: {text[:200]}") from e


def locate(png: bytes, target: str, transport: httpx.BaseTransport | None = None) -> dict | None:
    """Find `target` in a screenshot. Returns {top, bottom, left, right, confidence} as 0..1 fractions, or None."""
    system = ("You locate user-interface elements in screenshots. Return JSON "
              '{"found": bool, "box_2d": [ymin, xmin, ymax, xmax]} with coordinates normalised to 0-1000. '
              "If the element is not clearly visible, return found=false.")
    parts = [
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}},
        {"text": f"Element to find: {target}"},
    ]
    out = json.loads(generate(system, parts, json_mode=True, transport=transport))
    if not out.get("found") or len(out.get("box_2d") or []) != 4:
        return None
    y0, x0, y1, x1 = (max(0.0, min(1000.0, float(v))) / 1000 for v in out["box_2d"])
    return {"top": y0, "left": x0, "bottom": y1, "right": x1, "confidence": None}
