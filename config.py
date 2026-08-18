"""Load configuration from .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / ".openhouse" / "visitors.db"
LOG_DIR = Path(os.getenv("LOG_DIR", ".openhouse/logs"))

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
LLM_API_KEY = os.getenv("LLM_API_KEY", "nokey")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-0.5b-instruct")

# Face recognition
FACE_CAMERA_INDEX = int(os.getenv("FACE_CAMERA_INDEX", "0"))
FACE_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.5"))
FACE_JPEG_QUALITY = int(os.getenv("FACE_JPEG_QUALITY", "95"))

# Cooldown
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "45"))

# TTS
TTS_VOICE_INDEX = int(os.getenv("TTS_VOICE_INDEX", "0"))
TTS_RATE = int(os.getenv("TTS_RATE", "150"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
TTS_EDGE_VOICE = os.getenv("TTS_EDGE_VOICE", "en-US-JennyNeural")

# STT
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
STT_TIMEOUT = int(os.getenv("STT_TIMEOUT", "10"))
STT_PHONEME_THRESHOLD = float(os.getenv("STT_PHONEME_THRESHOLD", "0.6"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
