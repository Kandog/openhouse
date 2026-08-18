"""Text-to-speech using Edge neural voices."""

import asyncio
import os
import tempfile
import threading
import time
import edge_tts
import config

_speak_lock = threading.Lock()

VOICE = config.TTS_EDGE_VOICE


def _play_audio(path: str) -> None:
    import win32com.client

    player = win32com.client.Dispatch("WMPlayer.OCX")
    media = player.newMedia(path)
    player.currentMedia = media
    player.controls.play()

    while player.playState not in (1, 8, 10):
        time.sleep(0.1)

    player.close()


async def _speak_async(text: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name

    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(path)

    try:
        _play_audio(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def speak(text: str) -> None:
    if not text:
        return

    with _speak_lock:
        asyncio.run(_speak_async(text))
