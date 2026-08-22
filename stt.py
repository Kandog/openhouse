"""Speech-to-text for capturing visitor voice responses via microphone."""

import logging
import re
import time
import numpy as np
import speech_recognition as sr
try:
    import sounddevice as sd
except Exception as _sd_err:
    sd = None
import config

logger = logging.getLogger("openhouse")

_mic_muted = True  # Microphone is muted by default (during prompts, TTS, and LLM response)


def extract_name(text: str | None) -> str | None:
    """Extract a person's name from spoken phrase or clean speech text."""
    if not text:
        return None
    cleaned = text.strip().strip(".!?,")
    if not cleaned:
        return None

    # Common phrase patterns for identifying names
    patterns = [
        r"(?:my name is|i'm called|i am|i'm|call me|this is|it's|they call me)\s+([a-zA-Z\s'-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            words = extracted.split()
            if words:
                return " ".join(w.capitalize() for w in words[:2])

    # If no pattern matched, filter out noise/filler words
    words = cleaned.split()
    if len(words) <= 4:
        filtered = [w for w in words if w.lower() not in ("hi", "hello", "hey", "uh", "um", "yes", "no", "so", "is")]
        if filtered:
            return " ".join(w.capitalize() for w in filtered[:2])

    return " ".join(w.capitalize() for w in words[:2])


def is_mic_muted() -> bool:
    """Check if microphone is currently muted."""
    return _mic_muted


def set_mic_muted(muted: bool) -> None:
    """Mute or unmute the microphone explicitly."""
    global _mic_muted
    _mic_muted = muted
    logger.info("[stt] Microphone muted state changed to: %s", _mic_muted)


def _resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D int16 numpy array from orig_sr to target_sr using linear interpolation."""
    if orig_sr == target_sr:
        return audio_data
    if len(audio_data) == 0:
        return audio_data
    num_samples = int(round(len(audio_data) * target_sr / orig_sr))
    if num_samples == 0:
        return np.array([], dtype=np.int16)
    orig_indices = np.linspace(0, len(audio_data) - 1, num=len(audio_data))
    target_indices = np.linspace(0, len(audio_data) - 1, num=num_samples)
    resampled = np.interp(target_indices, orig_indices, audio_data.astype(np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def _get_input_sample_rate() -> int:
    """Query default input device samplerate or fallback to 16000/44100."""
    if sd is not None:
        try:
            device_info = sd.query_devices(kind="input")
            if device_info and "default_samplerate" in device_info and device_info["default_samplerate"]:
                sr = int(device_info["default_samplerate"])
                if sr > 0:
                    return sr
        except Exception as e:
            logger.debug("[stt] Could not query input device samplerate: %s", e)
    return 16000


def capture_name(timeout: int | None = None, phrase_time_limit: int | None = None) -> str | None:
    """Listen for a voice response and return the transcribed text, or None on failure.

    Microphone is unmuted ONLY during active audio capture in this function.
    Uses sounddevice to stream audio with dynamic energy thresholding and silence detection.
    Falls back gracefully if no speech is detected.
    """
    global _mic_muted
    try:
        set_mic_muted(False)

        recognizer = sr.Recognizer()

        timeout_val = timeout or getattr(config, "STT_TIMEOUT", 10)
        limit_val = phrase_time_limit or getattr(config, "STT_PHRASE_TIME_LIMIT", 8)

        device_sr = _get_input_sample_rate()
        target_sr = 16000

        logger.info(
            "[stt] Unmuted mic. Listening for visitor response (timeout: %ss, phrase limit: %ss, sample rate: %sHz)...",
            timeout_val,
            limit_val,
            device_sr,
        )

        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(device_sr * chunk_duration)

        max_silence_after_speech = 1.2  # Stop recording 1.2s after user stops talking
        max_pre_roll_chunks = 8  # Retain ~0.8s pre-roll audio buffer
        required_high_chunks = 2  # Require 2 consecutive high-energy chunks (200ms) to activate

        start_time = time.time()

        try:
            if sd is None:
                raise RuntimeError("sounddevice is not available")
            with sd.InputStream(samplerate=device_sr, channels=1, dtype="int16", blocksize=chunk_samples) as stream:
                # 1. Quick ambient calibration (0.3s)
                ambient_chunks = []
                for _ in range(3):
                    chunk, overflowed = stream.read(chunk_samples)
                    if not overflowed and len(chunk) > 0:
                        ambient_chunks.append(chunk)

                energy_threshold = 200.0
                if ambient_chunks:
                    ambient_data = np.concatenate(ambient_chunks, axis=0)
                    rms = float(np.sqrt(np.mean(ambient_data.astype(np.float64) ** 2)))
                    energy_threshold = min(max(150.0, rms * 1.5), 1000.0)
                    logger.debug("[stt] Calibrated ambient threshold: %.1f", energy_threshold)

                # 2. Continuous listening loop with retry
                while time.time() - start_time < timeout_val:
                    audio_frames = []
                    pre_roll_buffer = []
                    speech_started = False
                    speech_start_time = None
                    silence_duration = 0.0
                    high_energy_count = 0

                    while True:
                        now = time.time()
                        elapsed = now - start_time

                        if not speech_started and elapsed > timeout_val:
                            break

                        if speech_started and (now - speech_start_time) > limit_val:
                            logger.info("[stt] Reached maximum phrase limit (%ss)", limit_val)
                            break

                        chunk, _ = stream.read(chunk_samples)
                        if len(chunk) == 0:
                            continue

                        chunk_flat = chunk.flatten()
                        chunk_rms = float(np.sqrt(np.mean(chunk_flat.astype(np.float64) ** 2)))

                        if chunk_rms > energy_threshold:
                            high_energy_count += 1
                            if not speech_started:
                                pre_roll_buffer.append(chunk_flat)
                                if len(pre_roll_buffer) > max_pre_roll_chunks:
                                    pre_roll_buffer.pop(0)

                                if high_energy_count >= required_high_chunks:
                                    logger.info("[stt] Speech detected! Recording response...")
                                    speech_started = True
                                    speech_start_time = time.time()
                                    audio_frames.extend(pre_roll_buffer)
                                    pre_roll_buffer.clear()
                            else:
                                silence_duration = 0.0
                                audio_frames.append(chunk_flat)
                        else:
                            high_energy_count = 0
                            if speech_started:
                                silence_duration += chunk_duration
                                audio_frames.append(chunk_flat)
                                if silence_duration >= max_silence_after_speech:
                                    logger.info("[stt] End of speech detected (silence for %.1fs)", silence_duration)
                                    break
                            else:
                                pre_roll_buffer.append(chunk_flat)
                                if len(pre_roll_buffer) > max_pre_roll_chunks:
                                    pre_roll_buffer.pop(0)

                    if not audio_frames:
                        logger.info("[stt] Timeout: No speech detected within %ss", timeout_val)
                        break

                    # Combine recorded chunks into single audio array
                    raw_audio = np.concatenate(audio_frames, axis=0)

                    # Resample if device sample rate is not 16000 Hz
                    resampled_audio = _resample_audio(raw_audio, device_sr, target_sr)

                    # Convert to AudioData format for SpeechRecognition
                    audio_bytes = resampled_audio.tobytes()
                    audio = sr.AudioData(audio_bytes, target_sr, 2)

                    stt_lang = getattr(config, "STT_LANGUAGE", "en")

                    # 1. Try PocketSphinx (recognize_sphinx)
                    try:
                        text = recognizer.recognize_sphinx(audio, language=stt_lang).strip()
                        if text:
                            logger.info("[stt] Heard (via sphinx): %s", text)
                            return text
                    except sr.UnknownValueError:
                        logger.info("[stt] Speech was detected but could not be understood by sphinx.")
                    except Exception as sphinx_err:
                        logger.debug("[stt] Sphinx recognition error or unavailable: %s", sphinx_err)

                    # 2. Try Whisper local (recognize_whisper)
                    try:
                        text = recognizer.recognize_whisper(audio, model="base", language=stt_lang).strip()
                        if text:
                            logger.info("[stt] Heard (via whisper): %s", text)
                            return text
                    except sr.UnknownValueError:
                        logger.info("[stt] Speech was detected but could not be understood by whisper.")
                    except Exception as whisper_err:
                        logger.debug("[stt] Whisper recognition error or unavailable: %s", whisper_err)

                    # 3. Try Vosk local (recognize_vosk)
                    try:
                        text = recognizer.recognize_vosk(audio, language=stt_lang).strip()
                        if text:
                            logger.info("[stt] Heard (via vosk): %s", text)
                            return text
                    except sr.UnknownValueError:
                        logger.info("[stt] Speech was detected but could not be understood by vosk.")
                    except Exception as vosk_err:
                        logger.debug("[stt] Vosk recognition error or unavailable: %s", vosk_err)

                    logger.info("[stt] Audio clip did not produce recognizable text. Continuing to listen...")

        except Exception as stream_err:
            logger.warning("[stt] InputStream failed or not supported (%s), falling back to sd.rec", stream_err)
            if sd is not None:
                duration = min(timeout_val, limit_val)
                rec_data = sd.rec(int(device_sr * duration), samplerate=device_sr, channels=1, dtype=np.int16)
                sd.wait()
                if len(rec_data) > 0:
                    raw_audio = rec_data.flatten()
                    resampled_audio = _resample_audio(raw_audio, device_sr, target_sr)
                    audio = sr.AudioData(resampled_audio.tobytes(), target_sr, 2)
                    stt_lang = getattr(config, "STT_LANGUAGE", "en")
                    try:
                        text = recognizer.recognize_sphinx(audio, language=stt_lang).strip()
                        if text:
                            return text
                    except Exception:
                        pass
                    try:
                        text = recognizer.recognize_whisper(audio, model="base", language=stt_lang).strip()
                        if text:
                            return text
                    except Exception:
                        pass
                    try:
                        text = recognizer.recognize_vosk(audio, language=stt_lang).strip()
                        if text:
                            return text
                    except Exception:
                        pass

    except sr.WaitTimeoutError:
        logger.info("[stt] No speech detected within timeout.")
    except Exception as e:
        logger.error("[stt] Unexpected STT error: %s", e)
    finally:
        set_mic_muted(True)

    return None
