"""Language detection and Roman-Urdu normalisation.

The original utterance is never modified; `normalize` returns a separate string that is
only used for internal interpretation.
"""

import re
from dataclasses import dataclass

_URDU_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_WORD = re.compile(r"[a-zA-Z']+")

# Common Roman-Urdu spelling variants -> canonical form.
_VARIANTS: dict[str, str] = {
    "kro": "karo", "kar0": "karo", "karein": "karo", "kren": "karo", "karen": "karo",
    "kr": "kar", "mjhe": "mujhe", "mujhay": "mujhe", "mje": "mujhe", "muje": "mujhe",
    "bnao": "banao", "bana0": "banao", "bnado": "bana do", "banado": "bana do",
    "kholdo": "khol do", "khol0": "kholo",
    "mn": "mein", "hy": "hai", "hen": "hain",
    "pehlay": "pehle", "phle": "pehle", "pahle": "pehle",
    "phr": "phir", "fir": "phir",
    "dikhaao": "dikhao", "dkhao": "dikhao", "padho": "parho", "prho": "parho",
    "rukk": "ruk", "bs": "bas", "kia": "kya", "kiya": "kya", "yeh": "ye", "iss": "is",
    "websit": "website",
}
# Variants that are also English words; only applied when the utterance is not English.
_AMBIGUOUS_VARIANTS: dict[str, str] = {
    "mai": "mein", "main": "mein", "me": "mein", "he": "hai", "h": "hai", "or": "aur", "ar": "aur",
}

# Words that strongly indicate Roman Urdu when written in Latin script.
_ROMAN_URDU_MARKERS = {
    "karo", "kar", "mujhe", "banao", "bana", "do", "kholo", "khol", "mein", "hai", "hain", "pehle",
    "phir", "aur", "dikhao", "parho", "ruko", "ruk", "bas", "kya", "ye", "wo", "ko", "ki", "ka", "ke",
    "se", "nahi", "haan", "jo", "kaun", "kahan", "abhi", "chalao", "batao", "check", "theek", "acha",
}
# Ambiguous with English; only counted when other markers are present.
_WEAK = {"do", "is", "check", "ye", "ko", "se"}


@dataclass(frozen=True)
class LanguageInfo:
    language: str  # en | ur | roman_ur | mixed
    roman_urdu_ratio: float


def detect_language(text: str) -> LanguageInfo:
    if _URDU_SCRIPT.search(text):
        latin = _WORD.findall(text)
        return LanguageInfo("mixed" if len(latin) >= 2 else "ur", 1.0)
    words = [_VARIANTS.get(w.lower(), w.lower()) for w in _WORD.findall(text)]
    if not words:
        return LanguageInfo("en", 0.0)
    strong = sum(1 for w in words if w in _ROMAN_URDU_MARKERS and w not in _WEAK)
    weak = sum(1 for w in words if w in _WEAK)
    hits = strong + (weak if strong else 0)
    ratio = hits / len(words)
    if strong == 0 or ratio < 0.12:
        return LanguageInfo("en", ratio)
    return LanguageInfo("roman_ur" if ratio >= 0.5 else "mixed", ratio)


def normalize(text: str) -> str:
    """Lower-case, collapse whitespace and canonicalise Roman-Urdu spelling variants."""
    variants = _VARIANTS
    if detect_language(text).language != "en":
        variants = {**_VARIANTS, **_AMBIGUOUS_VARIANTS}
    out = []
    for token in re.split(r"(\s+)", text.strip()):
        low = token.lower()
        stripped = low.strip(".,!?;:")
        if stripped in variants:
            low = low.replace(stripped, variants[stripped])
        out.append(low)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def strip_wake_name(text: str) -> str:
    """Remove a leading "Abbas"/"Abbad" address from the utterance."""
    # Includes common speech-recognition mishearings of the names ("A bass", "A bad", "Abbot").
    names = r"abbas|abbad|abbaas|abaas|abad|abass|a\s+bass|a\s+bas|a\s+bad|a\s+bot|abbot|abbott|aba's|abbas'"
    return re.sub(rf"^\s*(hey\s+|ok\s+)?({names})\b[\s,.:!-]*", "", text, flags=re.I).strip()
