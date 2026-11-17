"""Speech-to-text adapter (faster-whisper, CPU int8 by default).

The browser records a push-to-talk clip; faster-whisper decodes it with PyAV and runs the
Silero VAD filter, so silence and non-speech are trimmed before recognition.
"""

import logging
import math
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent.config import get_settings

log = logging.getLogger("abbas.asr")


@dataclass
class Transcript:
    text: str
    language: str | None
    language_probability: float | None
    confidence: float
    duration_s: float
    elapsed_s: float
    segments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ModelMissing(RuntimeError):
    pass


class ASREngine(ABC):
    name: str

    @abstractmethod
    def status(self) -> dict: ...

    @abstractmethod
    def transcribe_file(self, path: str | Path) -> Transcript: ...

    @abstractmethod
    def transcribe_bytes(self, audio: bytes, suffix: str = ".webm") -> Transcript: ...


class WhisperASR(ASREngine):
    name = "faster-whisper"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    @property
    def download_root(self) -> Path:
        return get_settings().models_dir / "whisper"

    def status(self) -> dict:
        s = get_settings()
        info = {"engine": self.name, "model": s.asr_model, "device": s.asr_device,
                "compute_type": s.asr_compute_type, "loaded": self._model is not None}
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return {**info, "available": False, "setup": "uv sync --extra voice"}
        cached = any(self.download_root.glob(f"models--Systran--faster-whisper-{s.asr_model}*/snapshots/*/model.bin"))
        return {**info, "available": cached,
                "setup": None if cached else f"uv run python scripts/download_models.py asr --size {s.asr_model}"}

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            s = get_settings()
            try:
                self._model = WhisperModel(s.asr_model, device=s.asr_device, compute_type=s.asr_compute_type,
                                           download_root=str(self.download_root), local_files_only=True)
            except Exception as e:
                raise ModelMissing(f"Whisper model '{s.asr_model}' is not downloaded. Run: "
                                   f"uv run python scripts/download_models.py asr --size {s.asr_model}") from e
        return self._model

    def transcribe_file(self, path: str | Path) -> Transcript:
        s = get_settings()
        with self._lock:
            model = self._load()
            t0 = time.perf_counter()
            segments, info = model.transcribe(
                str(path), language=s.asr_language or None, beam_size=5, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                # Bias decoding toward the agent's names, otherwise "Abbas" is heard as "A bass".
                hotwords="Abbas Abbad", initial_prompt="Abbas, Abbad.",
            )
            segs = [{"start": round(x.start, 2), "end": round(x.end, 2), "text": x.text.strip(),
                     "avg_logprob": x.avg_logprob, "no_speech_prob": x.no_speech_prob} for x in segments]
            elapsed = time.perf_counter() - t0
        text = " ".join(x["text"] for x in segs).strip()
        if segs:
            # Duration-weighted mean token probability, discounted by the no-speech probability.
            total = sum(max(x["end"] - x["start"], 0.01) for x in segs)
            conf = sum(math.exp(x["avg_logprob"]) * (1 - x["no_speech_prob"]) * max(x["end"] - x["start"], 0.01)
                       for x in segs) / total
        else:
            conf = 0.0
        return Transcript(text=text, language=info.language, language_probability=round(info.language_probability, 3),
                          confidence=round(conf, 3), duration_s=round(info.duration, 2), elapsed_s=round(elapsed, 2),
                          segments=segs)

    def transcribe_bytes(self, audio: bytes, suffix: str = ".webm") -> Transcript:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"clip{suffix}"
            p.write_bytes(audio)
            return self.transcribe_file(p)  # raw audio is discarded with the temp dir


class ElevenLabsASR(ASREngine):
    """Adapter for ElevenLabs Scribe Speech-to-Text API."""

    name = "elevenlabs"

    def status(self) -> dict:
        s = get_settings()
        key = s.elevenlabs_api_key
        info = {
            "engine": self.name,
            "model": "scribe_v1",
            "device": "cloud",
            "compute_type": "api",
            "loaded": True,
        }
        if not key:
            return {**info, "available": False, "setup": "Set ABBAS_ELEVENLABS_API_KEY in .env"}
        return {**info, "available": True, "setup": None}

    def transcribe_bytes(self, audio: bytes, suffix: str = ".webm") -> Transcript:
        s = get_settings()
        if not s.elevenlabs_api_key:
            raise ModelMissing("ElevenLabs API key is missing. Set ABBAS_ELEVENLABS_API_KEY in .env")

        import httpx

        url = f"{s.elevenlabs_base_url.rstrip('/')}/speech-to-text"
        headers = {"xi-api-key": s.elevenlabs_api_key}

        mime_map = {
            ".webm": "audio/webm",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
        }
        mime_type = mime_map.get(suffix.lower(), "audio/webm")
        files = {"file": (f"clip{suffix}", audio, mime_type)}
        data = {"model_id": "scribe_v1"}
        if s.asr_language:
            data["language_code"] = s.asr_language

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, headers=headers, files=files, data=data)
                if resp.status_code != 200:
                    err_msg = resp.text
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("detail", {}).get("message", resp.text)
                    except Exception:
                        pass
                    raise RuntimeError(f"ElevenLabs STT error ({resp.status_code}): {err_msg}")
                res = resp.json()
        except httpx.RequestError as e:
            raise RuntimeError(f"ElevenLabs STT connection error: {e}") from e

        elapsed = time.perf_counter() - t0
        text = (res.get("text") or "").strip()
        words = [w for w in res.get("words", []) if w.get("type") == "word"]

        if words:
            word_probs = [math.exp(w.get("logprob", 0.0)) for w in words if "logprob" in w]
            conf = sum(word_probs) / len(word_probs) if word_probs else 0.95
        else:
            conf = 0.95 if text else 0.0

        # Build segments from words grouped by punctuation or natural pauses
        segments = []
        cur_words = []
        cur_start = None
        for w in res.get("words", []):
            if w.get("type") != "word":
                continue
            if cur_start is None:
                cur_start = w.get("start", 0.0)
            cur_words.append(w.get("text", "").strip())
            if any(w.get("text", "").endswith(p) for p in (".", "!", "?", "،", "؟")):
                segments.append({
                    "start": round(cur_start, 2),
                    "end": round(w.get("end", 0.0), 2),
                    "text": " ".join(cur_words),
                    "avg_logprob": w.get("logprob", 0.0),
                    "no_speech_prob": 0.0,
                })
                cur_words = []
                cur_start = None

        if cur_words and cur_start is not None:
            segments.append({
                "start": round(cur_start, 2),
                "end": round(res.get("audio_duration_secs", cur_start + 1.0), 2),
                "text": " ".join(cur_words),
                "avg_logprob": 0.0,
                "no_speech_prob": 0.0,
            })

        duration = round(res.get("audio_duration_secs", 0.0), 2)
        lang = res.get("language_code")
        lang_prob = round(res.get("language_probability", 1.0), 3) if res.get("language_probability") is not None else None

        return Transcript(
            text=text,
            language=lang,
            language_probability=lang_prob,
            confidence=round(conf, 3),
            duration_s=duration,
            elapsed_s=round(elapsed, 2),
            segments=segments,
        )

    def transcribe_file(self, path: str | Path) -> Transcript:
        p = Path(path)
        return self.transcribe_bytes(p.read_bytes(), suffix=p.suffix or ".wav")


class ASRDispatcher:
    def __init__(self) -> None:
        self._engines: dict[str, ASREngine] = {
            "whisper": WhisperASR(),
            "faster-whisper": WhisperASR(),
            "elevenlabs": ElevenLabsASR(),
        }

    def _engine(self) -> ASREngine:
        engine = (get_settings().asr_engine or "whisper").lower().replace("_", "").replace("-", "")
        if "eleven" in engine:
            return self._engines["elevenlabs"]
        return self._engines["whisper"]

    def status(self) -> dict:
        return self._engine().status()

    def transcribe_bytes(self, audio: bytes, suffix: str = ".webm") -> Transcript:
        return self._engine().transcribe_bytes(audio, suffix=suffix)

    def transcribe_file(self, path: str | Path) -> Transcript:
        return self._engine().transcribe_file(path)


asr = ASRDispatcher()
