"""Speech-to-text for capturing visitor voice responses via microphone."""

import logging
import numpy as np
import speech_recognition as sr
import sounddevice as sd
import config

logger = logging.getLogger("openhouse")

_mic_muted = True  # Microphone is muted by default (during prompts, TTS, and LLM response)


def is_mic_muted() -> bool:
    """Check if microphone is currently muted."""
    return _mic_muted


def set_mic_muted(muted: bool) -> None:
    """Mute or unmute the microphone explicitly."""
    global _mic_muted
    _mic_muted = muted
    logger.info("[stt] Microphone muted state changed to: %s", _mic_muted)


def capture_name(timeout: int | None = None, phrase_time_limit: int | None = None) -> str | None:
    """Listen for a voice response and return the transcribed text, or None on failure.

    Microphone is unmuted ONLY during active audio capture in this function.
    Uses Google Speech Recognition (free, requires internet).
    Uses sounddevice for audio capture.
    Falls back gracefully if no speech is detected.
    """
    global _mic_muted
    try:
        # Explicitly unmute microphone when starting audio capture
        set_mic_muted(False)

        recognizer = sr.Recognizer()

        timeout_val = timeout or getattr(config, "STT_TIMEOUT", 10)
        limit_val = phrase_time_limit or getattr(config, "STT_PHRASE_TIME_LIMIT", 5)
        duration = min(timeout_val, limit_val) if (timeout and phrase_time_limit) else (phrase_time_limit or timeout_val)

        logger.info("[stt] Unmuted mic. Listening for visitor response (duration: %ss)...", duration)

        sample_rate = 16000

        try:
            # Record audio from microphone
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
            sd.wait()

            # Mute microphone immediately after recording finishes
            set_mic_muted(True)

            # Convert to AudioData format for speech_recognition
            audio_bytes = audio_data.tobytes()
            audio = sr.AudioData(audio_bytes, sample_rate, 2)

            # Recognize speech using Google Speech Recognition
            stt_lang = getattr(config, "STT_LANGUAGE", "en")
            text = recognizer.recognize_google(audio, language=stt_lang).strip()
            if text:
                logger.info("[stt] Heard: %s", text)
                return text
        except Exception as e:
            logger.debug("[stt] Audio recording or recognition error: %s", e)

    except sr.WaitTimeoutError:
        logger.info("[stt] No speech detected within timeout.")
    except sr.UnknownValueError:
        logger.info("[stt] Could not understand audio.")
    except sr.RequestError as e:
        logger.warning("[stt] Speech recognition service error: %s", e)
    except Exception as e:
        logger.error("[stt] Unexpected STT error: %s", e)
    finally:
        # Always ensure microphone is muted when capture is complete or fails
        set_mic_muted(True)

    return None
