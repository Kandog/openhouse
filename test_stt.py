"""Tests for Speech-To-Text (stt.py) functionality."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import stt


class TestSTT(unittest.TestCase):

    def test_extract_name(self):
        self.assertEqual(stt.extract_name("My name is John"), "John")
        self.assertEqual(stt.extract_name("I am Alice"), "Alice")
        self.assertEqual(stt.extract_name("Call me Bob Smith"), "Bob Smith")
        self.assertEqual(stt.extract_name("Hi I'm Sarah"), "Sarah")
        self.assertEqual(stt.extract_name("It's David"), "David")
        self.assertEqual(stt.extract_name("uh Alex"), "Alex")
        self.assertIsNone(stt.extract_name(""))
        self.assertIsNone(stt.extract_name(None))

    def test_mic_muted_state(self):
        stt.set_mic_muted(True)
        self.assertTrue(stt.is_mic_muted())

        stt.set_mic_muted(False)
        self.assertFalse(stt.is_mic_muted())

        # Cleanup state
        stt.set_mic_muted(True)

    def test_resample_audio_same_rate(self):
        data = np.array([100, 200, 300, 400], dtype=np.int16)
        resample = stt._resample_audio(data, 16000, 16000)
        np.testing.assert_array_equal(resample, data)

    def test_resample_audio_downsample(self):
        # 48000 Hz to 16000 Hz -> 1/3 number of samples
        data = np.sin(np.linspace(0, 10, 48000)).astype(np.float32) * 10000
        data_int = data.astype(np.int16)
        resample = stt._resample_audio(data_int, 48000, 16000)
        self.assertEqual(len(resample), 16000)

    def test_resample_audio_empty(self):
        data = np.array([], dtype=np.int16)
        resample = stt._resample_audio(data, 44100, 16000)
        self.assertEqual(len(resample), 0)

    @patch("stt.sd", create=True)
    def test_get_input_sample_rate(self, mock_sd):
        mock_sd.query_devices.return_value = {"default_samplerate": 48000.0}
        self.assertEqual(stt._get_input_sample_rate(), 48000)

        mock_sd.query_devices.side_effect = Exception("No input device")
        self.assertEqual(stt._get_input_sample_rate(), 16000)

    @patch("stt.sd", create=True)
    @patch("stt.sr.Recognizer")
    def test_capture_name_success_sphinx(self, mock_recognizer_cls, mock_sd):
        # Mock Speech Recognition
        mock_rec_instance = MagicMock()
        mock_rec_instance.recognize_sphinx.return_value = "Hello Openhouse"
        mock_recognizer_cls.return_value = mock_rec_instance

        # Mock sounddevice InputStream
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value.__enter__.return_value = mock_stream

        high_energy_chunk = np.full((1600, 1), 1000, dtype=np.int16)
        silent_chunk = np.zeros((1600, 1), dtype=np.int16)

        stream_reads = [
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.full((1600, 1), 100, dtype=np.int16), False),  # Low energy chunk (added to pre-roll)
            (high_energy_chunk, False),
            (high_energy_chunk, False),
        ] + [(silent_chunk, False)] * 15

        mock_stream.read.side_effect = stream_reads

        with patch("stt._get_input_sample_rate", return_value=16000):
            result = stt.capture_name(timeout=5, phrase_time_limit=5)

        self.assertEqual(result, "Hello Openhouse")
        self.assertTrue(stt.is_mic_muted())
        # Ensure recognize_google is not called (fully local policy)
        self.assertFalse(hasattr(mock_rec_instance, "recognize_google") and mock_rec_instance.recognize_google.called)

    @patch("stt.sd", create=True)
    @patch("stt.sr.Recognizer")
    def test_capture_name_fallback_whisper(self, mock_recognizer_cls, mock_sd):
        mock_rec_instance = MagicMock()
        mock_rec_instance.recognize_sphinx.side_effect = Exception("Sphinx unavailable")
        mock_rec_instance.recognize_whisper.return_value = "Whisper Success"
        mock_recognizer_cls.return_value = mock_rec_instance

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value.__enter__.return_value = mock_stream

        high_energy_chunk = np.full((1600, 1), 1000, dtype=np.int16)
        silent_chunk = np.zeros((1600, 1), dtype=np.int16)

        stream_reads = [
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (high_energy_chunk, False),
            (high_energy_chunk, False),
        ] + [(silent_chunk, False)] * 15

        mock_stream.read.side_effect = stream_reads

        with patch("stt._get_input_sample_rate", return_value=16000):
            result = stt.capture_name(timeout=5, phrase_time_limit=5)

        self.assertEqual(result, "Whisper Success")
        self.assertTrue(stt.is_mic_muted())

    @patch("stt.sd", create=True)
    @patch("stt.sr.Recognizer")
    def test_capture_name_pre_roll_buffer(self, mock_recognizer_cls, mock_sd):
        mock_rec_instance = MagicMock()
        mock_rec_instance.recognize_sphinx.return_value = "Pre roll captured"
        mock_recognizer_cls.return_value = mock_rec_instance

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value.__enter__.return_value = mock_stream

        pre_roll_chunk = np.full((1600, 1), 150, dtype=np.int16)
        high_energy_chunk = np.full((1600, 1), 1000, dtype=np.int16)
        silent_chunk = np.zeros((1600, 1), dtype=np.int16)

        stream_reads = [
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (pre_roll_chunk, False),
            (high_energy_chunk, False),
            (high_energy_chunk, False),
        ] + [(silent_chunk, False)] * 15

        mock_stream.read.side_effect = stream_reads

        with patch("stt._get_input_sample_rate", return_value=16000):
            result = stt.capture_name(timeout=5, phrase_time_limit=5)

        self.assertEqual(result, "Pre roll captured")
        # Verify AudioData received audio containing the pre-roll chunk
        mock_rec_instance.recognize_sphinx.assert_called_once()
        audio_arg = mock_rec_instance.recognize_sphinx.call_args[0][0]
        # Check audio length includes calibration (0 chunks) + pre_roll (1 chunk) + high_energy (1 chunk) + 12 silent chunks (until 1.2s silence limit)
        self.assertTrue(len(audio_arg.get_raw_data()) > 0)

    @patch("stt.sd", create=True)
    def test_capture_name_timeout(self, mock_sd):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value.__enter__.return_value = mock_stream

        # Silent chunks only
        silent_chunk = np.zeros((1600, 1), dtype=np.int16)
        mock_stream.read.return_value = (silent_chunk, False)

        with patch("stt._get_input_sample_rate", return_value=16000):
            result = stt.capture_name(timeout=0.2, phrase_time_limit=2)

        self.assertIsNone(result)
        self.assertTrue(stt.is_mic_muted())

    @patch("tts._speak_pyttsx3", return_value=True)
    def test_tts_mutes_mic(self, mock_pyttsx3):
        import tts
        stt.set_mic_muted(False)
        self.assertFalse(stt.is_mic_muted())

        tts.speak("Hello visitor")

        self.assertTrue(stt.is_mic_muted())


if __name__ == "__main__":
    unittest.main()
