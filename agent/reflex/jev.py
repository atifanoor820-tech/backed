"""Thin client for TypeSafe's Jev (POST /v1/systemone).

Jev returns typed answers with calibrated probabilities instead of text:
  noul   -> P(yes)
  choice -> winning option, confidence, per-option probabilities (options fixed up front)
  score  -> probability-weighted level
  box    -> bounding box in an image asset (experimental, needs the Danger-Zone-Images header)
"""

import base64
import json
import logging

import httpx

from agent.config import get_settings

log = logging.getLogger("abbas.jev")


class JevError(RuntimeError):
    pass


def ask(state, questions: dict, assets: dict | None = None, transport: httpx.BaseTransport | None = None) -> dict:
    s = get_settings()
    if not s.jev_enabled:
        raise JevError("Jev is not configured (set ABBAS_JEV_API_KEY)")
    body = {"model": s.jev_model, "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {s.jev_api_key}"}
    if assets:
        body["assets"] = assets
        headers["TypeSafe-Danger-Zone-Images"] = "true"
    if len(json.dumps(state)) > 100_000:
        headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
    with httpx.Client(timeout=20, transport=transport) as c:
        r = c.post(f"{s.jev_base_url.rstrip('/')}/systemone", json=body, headers=headers)
    if r.status_code != 200:
        raise JevError(f"Jev HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def box(png: bytes, target: str, transport: httpx.BaseTransport | None = None) -> dict | None:
    out = ask(
        state=f"A screenshot of a Windows desktop. Find: {target}",
        questions={"target": {"type": "box", "instructions": f"Bounding box of: {target}"}},
        assets={"screen": {"type": "image", "format": "png", "data_base64": base64.b64encode(png).decode()}},
        transport=transport,
    )
    ans = out["answers"]["target"]
    b = ans.get("box")
    return {**b, "confidence": b.get("confidence", ans.get("confidence"))} if b else None
