"""Tests for Speech-To-Text (stt.py) functionality."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import stt


class TestSTT(unittest.TestCase):

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

    @patch("stt.sd.query_devices")
    def test_get_input_sample_rate(self, mock_query):
        mock_query.return_value = {"default_samplerate": 48000.0}
        self.assertEqual(stt._get_input_sample_rate(), 48000)

        mock_query.side_effect = Exception("No input device")
        self.assertEqual(stt._get_input_sample_rate(), 16000)

    @patch("stt.sd.InputStream")
    @patch("stt.sr.Recognizer")
    def test_capture_name_success(self, mock_recognizer_cls, mock_input_stream_cls):
        # Mock Speech Recognition
        mock_rec_instance = MagicMock()
        mock_rec_instance.recognize_google.return_value = "Hello Openhouse"
        mock_recognizer_cls.return_value = mock_rec_instance

        # Mock sounddevice InputStream
        mock_stream = MagicMock()
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream

        # Speech chunk with high energy
        high_energy_chunk = np.full((1600, 1), 1000, dtype=np.int16)
        silent_chunk = np.zeros((1600, 1), dtype=np.int16)

        # Ambient noise calibration (3 chunks), speech (1 chunk), silence (12 chunks -> 1.2s silence to trigger end of speech)
        stream_reads = [
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (high_energy_chunk, False),
        ] + [(silent_chunk, False)] * 15

        mock_stream.read.side_effect = stream_reads

        with patch("stt._get_input_sample_rate", return_value=16000):
            result = stt.capture_name(timeout=5, phrase_time_limit=5)

        self.assertEqual(result, "Hello Openhouse")
        self.assertTrue(stt.is_mic_muted())

    @patch("stt.sd.InputStream")
    def test_capture_name_timeout(self, mock_input_stream_cls):
        mock_stream = MagicMock()
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream

        # Silent chunks only
        silent_chunk = np.zeros((1600, 1), dtype=np.int16)
        mock_stream.read.return_value = (silent_chunk, False)

        with patch("stt._get_input_sample_rate", return_value=16000):
            result = stt.capture_name(timeout=0.2, phrase_time_limit=2)

        self.assertIsNone(result)
        self.assertTrue(stt.is_mic_muted())

    @patch("stt.sd.InputStream")
    @patch("stt.sr.Recognizer")
    def test_capture_name_retry_on_network_error(self, mock_recognizer_cls, mock_input_stream_cls):
        # Mock Speech Recognition with initial connection error then success
        mock_rec_instance = MagicMock()
        mock_rec_instance.recognize_google.side_effect = [
            OSError("[WinError 10054] An existing connection was forcibly closed"),
            "Retry Success",
        ]
        mock_recognizer_cls.return_value = mock_rec_instance

        # Mock sounddevice InputStream
        mock_stream = MagicMock()
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream

        high_energy_chunk = np.full((1600, 1), 1000, dtype=np.int16)
        silent_chunk = np.zeros((1600, 1), dtype=np.int16)

        stream_reads = [
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (np.zeros((1600, 1), dtype=np.int16), False),
            (high_energy_chunk, False),
        ] + [(silent_chunk, False)] * 15

        mock_stream.read.side_effect = stream_reads

        with patch("stt._get_input_sample_rate", return_value=16000):
            result = stt.capture_name(timeout=5, phrase_time_limit=5)

        self.assertEqual(result, "Retry Success")
        self.assertEqual(mock_rec_instance.recognize_google.call_count, 2)
        self.assertTrue(stt.is_mic_muted())


if __name__ == "__main__":
    unittest.main()
