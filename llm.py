"""Generate personalized greetings using a local LLM via Ollama."""

import requests
import config


_NEW_PROMPT = """\
You are a warm, friendly host at an openhouse event.
A new visitor has just arrived. Generate a short, personal welcome message.
Use their name naturally. Keep it under 20 words. Do not add any preamble or explanation — just the message to speak aloud.
Visitor name: {name}
"""

_RETURN_PROMPT = """\
You are a warm, friendly host at an openhouse event.
A returning visitor has come back. Generate a short, personal welcome-back message.
Make it feel genuinely warm and recognisable — like you remember them.
Keep it under 20 words. Do not add any preamble or explanation — just the message to speak aloud.
Visitor name: {name}
Last visited: {last_seen}
"""


def _call_llm(prompt: str) -> str:
    payload = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{config.LLM_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        # LLM Studio returns OpenAI-compatible format
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        return ""
    except Exception as e:
        print(f"[llm] Error calling LLM: {e}")
        return ""


def generate_chat_response(prompt: str) -> str:
    """Generate a chat response for the interactive text/speech input."""
    chat_prompt = (
        "You are a warm, friendly host at an openhouse event. "
        f"User says: {prompt}. "
        "Keep your reply short (under 30 words), helpful, and conversational. "
        "Do not add any preamble or explanation — just speak the message directly."
    )
    try:
        return _call_llm(chat_prompt)
    except Exception:
        return "Welcome! Let me know how I can help you today."


def generate_new_visitor_greeting(name: str) -> str:
    prompt = _NEW_PROMPT.format(name=name)
    try:
        return _call_llm(prompt)
    except Exception:
        return f"Hello {name}! Welcome to our openhouse!"


def generate_return_greeting(name: str, last_seen: str) -> str:
    prompt = _RETURN_PROMPT.format(name=name, last_seen=last_seen)
    try:
        return _call_llm(prompt)
    except Exception:
        return f"Welcome back! Great to see you again at our openhouse!"
