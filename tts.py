"""Text-to-speech module for Openhouse AI Assistant.

Supports pyttsx3 (offline Windows SAPI5) and edge-tts (online neural voice)
with reliable playback and fallback mechanisms.
"""

import os
import sys
import tempfile
import threading
import time
import logging
import config
import stt

logger = logging.getLogger("openhouse")
_speak_lock = threading.Lock()


def _speak_pyttsx3(text: str) -> bool:
    """Speak text using pyttsx3 with a fresh engine instance per call."""
    try:
        import pyttsx3
        engine = pyttsx3.init()

        # Configure rate & volume
        rate = getattr(config, "TTS_RATE", 150)
        volume = getattr(config, "TTS_VOLUME", 1.0)
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        # Configure voice index if valid
        voices = engine.getProperty("voices")
        voice_index = getattr(config, "TTS_VOICE_INDEX", 0)
        if voices and 0 <= voice_index < len(voices):
            engine.setProperty("voice", voices[voice_index].id)

        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning("[tts] pyttsx3 failed: %s", e)
        return False


def _play_audio_windows(path: str) -> bool:
    """Play audio file on Windows using WMPlayer.OCX or MCI / powershell."""
    played = False
    try:
        import win32com.client
        player = win32com.client.Dispatch("WMPlayer.OCX")
        media = player.newMedia(path)
        player.currentMedia = media
        player.controls.play()

        # Wait for media to start loading/playing
        for _ in range(50):
            time.sleep(0.05)
            # 3 = Playing, 6 = Buffering, 9 = Transitioning
            if player.playState in (3, 6, 9):
                break

        # Wait while playing or buffering or transitioning
        while player.playState in (3, 6, 9):
            time.sleep(0.1)

        played = True
        try:
            player.close()
        except Exception:
            pass
    except Exception as e:
        logger.warning("[tts] WMPlayer.OCX play failed: %s", e)

    if not played:
        # Fallback to PowerShell audio playback on Windows
        try:
            import subprocess
            cmd = f'powershell -c "(New-Object Media.SoundPlayer \'{path}\').PlaySync()"'
            subprocess.run(cmd, shell=True, check=True)
            played = True
        except Exception as e:
            logger.warning("[tts] PowerShell SoundPlayer failed: %s", e)

    return played


def _speak_edge_tts(text: str) -> bool:
    """Speak text using edge-tts."""
    try:
        import asyncio
        import edge_tts

        voice = getattr(config, "TTS_EDGE_VOICE", "en-US-JennyNeural")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            path = f.name

        async def _gen():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(path)

        asyncio.run(_gen())

        if os.path.exists(path) and os.path.getsize(path) > 0:
            success = _play_audio_windows(path)
            try:
                os.remove(path)
            except OSError:
                pass
            return success
        return False
    except Exception as e:
        logger.warning("[tts] edge-tts failed: %s", e)
        return False


def speak(text: str) -> None:
    """Speak the given text aloud. Try pyttsx3 first, fallback to edge-tts if needed."""
    if not text or not text.strip():
        return

    text = text.strip()
    logger.info("[tts] Speaking: %s", text)

    # Ensure microphone is muted during any system response to speaker
    stt.set_mic_muted(True)

    with _speak_lock:
        # Try pyttsx3 first (fast, local, offline Windows SAPI5)
        if _speak_pyttsx3(text):
            return

        # Fallback to edge-tts
        if _speak_edge_tts(text):
            return

        logger.error("[tts] All TTS methods failed to speak: %s", text)
