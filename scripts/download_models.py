"""Download local voice models into data/models (kept separate from app startup).

    uv run python scripts/download_models.py asr --size base
    uv run python scripts/download_models.py tts
"""

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import get_settings  # noqa: E402

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


def download_asr(size: str) -> None:
    from faster_whisper import WhisperModel

    root = get_settings().models_dir / "whisper"
    print(f"Downloading faster-whisper '{size}' (Systran, MIT) to {root} ...")
    WhisperModel(size, device="cpu", compute_type="int8", download_root=str(root))
    print("ASR model ready.")


def download_tts() -> None:
    folder = get_settings().models_dir / "kokoro"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        dest = folder / name
        if dest.exists():
            print(f"{name} already present")
            continue
        print(f"Downloading {name} (Kokoro-82M, Apache-2.0) ...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(KOKORO_BASE + name, tmp)
        tmp.rename(dest)
    print("TTS model ready.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["asr", "tts", "all"])
    ap.add_argument("--size", default=get_settings().asr_model)
    a = ap.parse_args()
    if a.what in ("asr", "all"):
        download_asr(a.size)
    if a.what in ("tts", "all"):
        download_tts()
