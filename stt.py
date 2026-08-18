"""Speech-to-text for capturing a visitor's name via microphone."""

import speech_recognition as sr
import sounddevice as sd
import numpy as np
import config


def capture_name(timeout: int | None = None, phrase_time_limit: int | None = None) -> str | None:
    """Listen for a voice response and return the transcribed text, or None on failure.

    Uses Google Speech Recognition (free, requires internet).
    Uses sounddevice for audio capture (works with Python 3.14+).
    Falls back gracefully if no speech is detected.
    """
    try:
        recognizer = sr.Recognizer()

        timeout = timeout or config.STT_TIMEOUT
        phrase_time_limit = phrase_time_limit or min(timeout, 8)

        print("[stt] Listening for your response...")

        # Record audio using sounddevice
        sample_rate = 16000
        duration = timeout

        try:
            # Record audio from microphone
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
            sd.wait()

            # Convert to AudioData format for speech_recognition
            audio_bytes = audio_data.tobytes()
            audio = sr.AudioData(audio_bytes, sample_rate, 2)

            # Recognize speech using Google
            text = recognizer.recognize_google(audio, language=config.STT_LANGUAGE).strip()
            if text:
                print(f"[stt] Heard: {text}")
                return text
        except Exception as e:
            print(f"[stt] Audio recording error: {e}")

    except sr.WaitTimeoutError:
        print("[stt] No speech detected within timeout.")
    except sr.UnknownValueError:
        print("[stt] Could not understand audio.")
    except sr.RequestError as e:
        print(f"[stt] Speech recognition service error: {e}")
    except Exception as e:
        print(f"[stt] Unexpected error: {e}")

    return None
