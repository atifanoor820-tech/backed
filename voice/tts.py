"""Text-to-speech adapters behind one interface. Returns WAV bytes for the dashboard to play.

Kokoro (kokoro-onnx, Apache-2.0 weights) is the default. Its languages are English, Spanish, French,
Hindi, Italian, Japanese, Portuguese and Chinese - it has no Urdu voice, so Roman-Urdu replies are
spoken with the configured English voice and will sound accented.
"""

import io
import logging
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from agent.config import get_settings

log = logging.getLogger("abbas.tts")


class TTSUnavailable(RuntimeError):
    pass


def speakable(text: str, limit: int = 600) -> str:
    """Strip symbols that read badly aloud and keep the spoken reply short."""
    text = re.sub(r"[✔✘…*`#>|]", " ", text)
    text = re.sub(r"https?://\S+", "the link", text)
    text = re.sub(r"[A-Za-z]:\\\S+", "the file", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


class TTSEngine(ABC):
    name: str

    @abstractmethod
    def status(self) -> dict: ...

    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...


def _wav(samples, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class KokoroTTS(TTSEngine):
    name = "kokoro"
    MODEL = "kokoro-v1.0.onnx"
    VOICES = "voices-v1.0.bin"

    def __init__(self) -> None:
        self._k = None
        self._lock = threading.Lock()

    @property
    def folder(self) -> Path:
        return get_settings().models_dir / "kokoro"

    def status(self) -> dict:
        s = get_settings()
        info = {"engine": self.name, "voice": s.tts_voice, "speed": s.tts_speed, "loaded": self._k is not None,
                "languages": ["en-us", "en-gb", "es", "fr-fr", "hi", "it", "ja", "pt-br", "cmn"]}
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError:
            return {**info, "available": False, "setup": "uv sync --extra voice"}
        ok = (self.folder / self.MODEL).exists() and (self.folder / self.VOICES).exists()
        return {**info, "available": ok, "setup": None if ok else "uv run python scripts/download_models.py tts"}

    def synthesize(self, text: str) -> bytes:
        s = get_settings()
        with self._lock:
            if self._k is None:
                if not self.status()["available"]:
                    raise TTSUnavailable("Kokoro model files missing. Run: uv run python scripts/download_models.py tts")
                from kokoro_onnx import Kokoro

                self._k = Kokoro(str(self.folder / self.MODEL), str(self.folder / self.VOICES))
            lang = "en-gb" if s.tts_voice.startswith("b") else "en-us"
            samples, sr = self._k.create(text, voice=s.tts_voice, speed=s.tts_speed, lang=lang)
        return _wav(samples, sr)


class PiperTTS(TTSEngine):
    """Adapter for piper1-gpl (`pip install piper-tts`, GPL-3.0). Not installed by default."""

    name = "piper"

    def status(self) -> dict:
        path = get_settings().piper_model_path
        try:
            import piper  # noqa: F401
        except ImportError:
            return {"engine": self.name, "available": False,
                    "setup": "uv pip install piper-tts and set ABBAS_PIPER_MODEL_PATH to a .onnx voice"}
        return {"engine": self.name, "available": bool(path) and Path(path).exists(), "model": path or None,
                "setup": None if path and Path(path).exists() else "set ABBAS_PIPER_MODEL_PATH"}

    def synthesize(self, text: str) -> bytes:
        if not self.status()["available"]:
            raise TTSUnavailable(self.status()["setup"])
        import wave

        from piper import PiperVoice

        voice = PiperVoice.load(get_settings().piper_model_path)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        return buf.getvalue()


class ElevenLabsTTS(TTSEngine):
    """Adapter for ElevenLabs Text-to-Speech API.
    Supports free plan premade voices (George, Roger, Sarah, Alice, Charlie, etc.).
    """

    name = "elevenlabs"

    PREMADE_VOICES: dict[str, str] = {
        "george": "JBFqnCBsd6RMkjVDRZzb",
        "roger": "CwhRBWXzGAHq8TQ4Fs17",
        "sarah": "EXAVITQu4vr4xnSDxMaL",
        "alice": "Xb7hH8MSUJpSbSDYk0k2",
        "charlie": "IKne3meq5aSn9XLyUdCD",
        "callum": "N2lVS1w4EtoT3dr4eOWO",
        "river": "SAz9YHcvj6GT2YYXdXww",
        "liam": "TX3LPaxmHKxFdv7VOQHJ",
        "laura": "FGY2WhTYpPnrIDTdsKH5",
        "harry": "SOYHLrjzK2X1ezoPC6cr",
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "adam": "pNInz6obpgDQGcFmaJgB",
    }

    def _resolve_voice_id(self, voice_setting: str) -> str:
        s = get_settings()
        v = (voice_setting or s.elevenlabs_voice_id or "").strip()
        lower_v = v.lower()
        if lower_v in self.PREMADE_VOICES:
            return self.PREMADE_VOICES[lower_v]
        if v:
            return v
        return self.PREMADE_VOICES["george"]

    def status(self) -> dict:
        s = get_settings()
        key = s.elevenlabs_api_key
        voice_id = self._resolve_voice_id(s.tts_voice)
        info = {
            "engine": self.name,
            "voice": s.tts_voice,
            "voice_id": voice_id,
            "speed": s.tts_speed,
            "model": s.elevenlabs_model_id,
        }
        if not key:
            return {**info, "available": False, "setup": "Set ABBAS_ELEVENLABS_API_KEY in .env"}
        return {**info, "available": True, "setup": None}

    def synthesize(self, text: str) -> bytes:
        s = get_settings()
        if not s.elevenlabs_api_key:
            raise TTSUnavailable("ElevenLabs API key is missing. Set ABBAS_ELEVENLABS_API_KEY in .env")

        import wave
        import httpx

        voice_id = self._resolve_voice_id(s.tts_voice)
        url = f"{s.elevenlabs_base_url.rstrip('/')}/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": s.elevenlabs_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": s.elevenlabs_model_id or "eleven_multilingual_v2",
        }
        params = {"output_format": "pcm_16000"}

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload, params=params)
                if resp.status_code != 200:
                    err_msg = resp.text
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("detail", {}).get("message", resp.text)
                    except Exception:
                        pass
                    raise TTSUnavailable(f"ElevenLabs TTS failed ({resp.status_code}): {err_msg}")
                raw_pcm = resp.content
        except httpx.RequestError as e:
            raise TTSUnavailable(f"ElevenLabs connection failed: {e}") from e

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_pcm)
        return buf.getvalue()


class NoTTS(TTSEngine):
    name = "none"

    def status(self) -> dict:
        return {"engine": self.name, "available": False, "setup": "set ABBAS_TTS_ENGINE=elevenlabs, kokoro or piper"}

    def synthesize(self, text: str) -> bytes:
        raise TTSUnavailable("Text-to-speech is disabled (ABBAS_TTS_ENGINE=none)")


_ENGINES: dict[str, TTSEngine] = {
    "kokoro": KokoroTTS(),
    "piper": PiperTTS(),
    "elevenlabs": ElevenLabsTTS(),
    "none": NoTTS(),
}


def get_tts() -> TTSEngine:
    return _ENGINES.get(get_settings().tts_engine, _ENGINES["none"])
