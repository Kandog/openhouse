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

try:
    from faster_whisper import WhisperModel
except Exception as _fw_err:
    WhisperModel = None

import config

logger = logging.getLogger("openhouse")

_mic_muted = True  # Microphone is muted by default (during prompts, TTS, and LLM response)
_fw_model = None


def _get_faster_whisper_model():
    """Lazy initialization of faster-whisper tiny model."""
    global _fw_model
    if _fw_model is None and WhisperModel is not None:
        try:
            logger.info("[stt] Initializing faster-whisper tiny model...")
            _fw_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception as e:
            logger.error("[stt] Failed to initialize faster-whisper model: %s", e)
    return _fw_model


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

        audio_frames = []
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(device_sr * chunk_duration)

        # Voice activity detection parameters
        speech_started = False
        start_time = time.time()
        speech_start_time = None
        silence_duration = 0.0
        max_silence_after_speech = 1.5  # Stop recording 1.5s after user stops talking

        # Pre-roll buffer to retain lead-in audio (up to ~0.8s) before trigger
        pre_roll_buffer = []
        max_pre_roll_chunks = 8

        # Collect ambient noise baseline for 0.3s if possible, or set static energy threshold
        energy_threshold = 100.0

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
                    # Cap initial threshold to ensure soft speech is reliably detected
                    energy_threshold = min(max(50.0, rms * 1.3), 1000.0)
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
                            # Include pre-roll buffer when speech starts to capture initial consonant/syllable
                            audio_frames.extend(pre_roll_buffer)
                            pre_roll_buffer.clear()
                        silence_duration = 0.0
                        audio_frames.append(chunk_flat)
                    else:
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

        # 1. Try faster-whisper tiny model
        fw_model = _get_faster_whisper_model()
        if fw_model is not None:
            try:
                audio_float32 = resampled_audio.astype(np.float32) / 32768.0
                segments, _ = fw_model.transcribe(audio_float32, language=stt_lang)
                text = "".join([s.text for s in segments]).strip()
                if text:
                    logger.info("[stt] Heard (via faster-whisper): %s", text)
                    return text
            except Exception as fw_err:
                logger.debug("[stt] faster-whisper recognition error or unavailable: %s", fw_err)

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
