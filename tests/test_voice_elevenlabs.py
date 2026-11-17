from unittest.mock import MagicMock, patch
import pytest

from agent.config import get_settings
from voice.asr import ElevenLabsASR, ASRDispatcher, Transcript, ModelMissing
from voice.tts import ElevenLabsTTS, TTSUnavailable


def test_elevenlabs_tts_voice_resolution():
    tts = ElevenLabsTTS()
    assert tts._resolve_voice_id("george") == "JBFqnCBsd6RMkjVDRZzb"
    assert tts._resolve_voice_id("Roger") == "CwhRBWXzGAHq8TQ4Fs17"
    assert tts._resolve_voice_id("sarah") == "EXAVITQu4vr4xnSDxMaL"
    assert tts._resolve_voice_id("alice") == "Xb7hH8MSUJpSbSDYk0k2"
    assert tts._resolve_voice_id("custom_voice_123") == "custom_voice_123"


def test_elevenlabs_tts_status_without_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "elevenlabs_api_key", "")
    tts = ElevenLabsTTS()
    st = tts.status()
    assert st["available"] is False
    assert "ABBAS_ELEVENLABS_API_KEY" in st["setup"]


def test_elevenlabs_tts_synthesize_mock():
    tts = ElevenLabsTTS()
    # 4 bytes of dummy PCM audio
    dummy_pcm = b"\x00\x00\x10\x00"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = dummy_pcm

    with patch("httpx.Client.post", return_value=mock_resp):
        wav_bytes = tts.synthesize("Hello world")
        assert wav_bytes.startswith(b"RIFF")
        assert b"WAVE" in wav_bytes[:12]


def test_elevenlabs_tts_error_handling():
    tts = ElevenLabsTTS()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    mock_resp.json.return_value = {"detail": {"message": "Invalid API key"}}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(TTSUnavailable) as exc:
            tts.synthesize("Hello")
        assert "Invalid API key" in str(exc.value)


def test_elevenlabs_asr_status_without_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "elevenlabs_api_key", "")
    asr_engine = ElevenLabsASR()
    st = asr_engine.status()
    assert st["available"] is False
    assert "ABBAS_ELEVENLABS_API_KEY" in st["setup"]


def test_elevenlabs_asr_transcribe_mock():
    asr_engine = ElevenLabsASR()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "text": "Open calculator",
        "language_code": "eng",
        "language_probability": 0.98,
        "audio_duration_secs": 1.5,
        "words": [
            {"text": "Open", "start": 0.0, "end": 0.5, "type": "word", "logprob": -0.01},
            {"text": "calculator.", "start": 0.5, "end": 1.2, "type": "word", "logprob": -0.02},
        ],
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        tr = asr_engine.transcribe_bytes(b"dummy audio", suffix=".wav")
        assert isinstance(tr, Transcript)
        assert tr.text == "Open calculator"
        assert tr.language == "eng"
        assert tr.duration_s == 1.5
        assert len(tr.segments) == 1
        assert tr.segments[0]["text"] == "Open calculator."


def test_asr_dispatcher(monkeypatch):
    dispatcher = ASRDispatcher()
    monkeypatch.setattr(get_settings(), "asr_engine", "elevenlabs")
    assert dispatcher._engine().name == "elevenlabs"

    monkeypatch.setattr(get_settings(), "asr_engine", "whisper")
    assert dispatcher._engine().name == "faster-whisper"
