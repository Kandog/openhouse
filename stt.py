"""Speech-to-text for capturing visitor voice responses via microphone."""

import logging
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

        audio_frames = []
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(device_sr * chunk_duration)

        # Voice activity detection parameters
        speech_started = False
        start_time = time.time()
        speech_start_time = None
        silence_duration = 0.0
        max_silence_after_speech = 1.2  # Stop recording 1.2s after user stops talking

        # Collect ambient noise baseline for 0.3s if possible, or set static energy threshold
        energy_threshold = 300.0

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

                if ambient_chunks:
                    ambient_data = np.concatenate(ambient_chunks, axis=0)
                    rms = float(np.sqrt(np.mean(ambient_data.astype(np.float64) ** 2)))
                    energy_threshold = max(300.0, rms * 2.5)
                    logger.debug("[stt] Calibrated ambient threshold: %.1f", energy_threshold)

                # 2. Streaming loop with VAD
                while True:
                    now = time.time()
                    elapsed = now - start_time

                    if not speech_started and elapsed > timeout_val:
                        logger.info("[stt] Timeout: No speech detected within %ss", timeout_val)
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
                        if not speech_started:
                            logger.info("[stt] Speech detected! Recording response...")
                            speech_started = True
                            speech_start_time = time.time()
                        silence_duration = 0.0
                        audio_frames.append(chunk_flat)
                    else:
                        if speech_started:
                            silence_duration += chunk_duration
                            audio_frames.append(chunk_flat)
                            if silence_duration >= max_silence_after_speech:
                                logger.info("[stt] End of speech detected (silence for %.1fs)", silence_duration)
                                break
        except Exception as stream_err:
            logger.warning("[stt] InputStream failed or not supported (%s), falling back to sd.rec", stream_err)
            if sd is not None:
                duration = min(timeout_val, limit_val)
                rec_data = sd.rec(int(device_sr * duration), samplerate=device_sr, channels=1, dtype=np.int16)
                sd.wait()
                if len(rec_data) > 0:
                    audio_frames = [rec_data.flatten()]

        set_mic_muted(True)

        if not audio_frames:
            logger.info("[stt] No audio captured.")
            return None

        # Combine recorded chunks into single audio array
        raw_audio = np.concatenate(audio_frames, axis=0)

        # Resample if device sample rate is not 16000 Hz
        resampled_audio = _resample_audio(raw_audio, device_sr, target_sr)

        # Convert to AudioData format for SpeechRecognition
        audio_bytes = resampled_audio.tobytes()
        audio = sr.AudioData(audio_bytes, target_sr, 2)

        # Perform local speech recognition (offline only, no Google or remote API)
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

    except sr.WaitTimeoutError:
        logger.info("[stt] No speech detected within timeout.")
    except Exception as e:
        logger.error("[stt] Unexpected STT error: %s", e)
    finally:
        set_mic_muted(True)

    return None
