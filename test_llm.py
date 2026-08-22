"""Tests for LLM response generation and length constraints (llm.py)."""

import unittest
from unittest.mock import patch
import llm


class TestLLM(unittest.TestCase):

    def test_trim_to_word_limit_short(self):
        text = "Hello! Welcome to our open house!"
        trimmed = llm._trim_to_word_limit(text, max_words=30)
        self.assertEqual(trimmed, text)
        self.assertLess(len(trimmed.split()), 35)

    def test_trim_to_word_limit_long(self):
        long_text = "word " * 50
        trimmed = llm._trim_to_word_limit(long_text, max_words=30)
        word_count = len(trimmed.split())
        self.assertLess(word_count, 35)
        self.assertLessEqual(word_count, 30)

    @patch("llm._call_llm")
    def test_generate_chat_response_length(self, mock_call):
        # Return a response longer than 35 words
        mock_call.return_value = " ".join([f"word{i}" for i in range(50)])
        resp = llm.generate_chat_response("Tell me about the house")
        self.assertLess(len(resp.split()), 35)

    @patch("llm._call_llm")
    def test_generate_new_visitor_greeting_length(self, mock_call):
        mock_call.return_value = " ".join([f"word{i}" for i in range(40)])
        resp = llm.generate_new_visitor_greeting("Alice")
        self.assertLess(len(resp.split()), 35)

    @patch("llm._call_llm")
    def test_generate_return_greeting_length(self, mock_call):
        mock_call.return_value = " ".join([f"word{i}" for i in range(45)])
        resp = llm.generate_return_greeting("Bob", "yesterday")
        self.assertLess(len(resp.split()), 35)


if __name__ == "__main__":
    unittest.main()
