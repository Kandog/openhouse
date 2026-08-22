"""Generate personalized greetings and response using a local LLM via OpenAI API endpoint.

References local openhouse Markdown documentation to answer property questions.
"""

import os
import requests
import config

def _get_openhouse_doc() -> str:
    """Read Openhouse Markdown reference documentation if available."""
    try:
        doc_path = getattr(config, "OPENHOUSE_DOC_PATH", None)
        if doc_path and os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
    except Exception as e:
        print(f"[llm] Error reading openhouse doc: {e}")
    return ""


def _trim_to_word_limit(text: str, max_words: int = 30) -> str:
    """Ensure response text strictly contains fewer than 35 words (at most max_words)."""
    if not text:
        return text
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words)

    truncated_words = words[:max_words]
    result = " ".join(truncated_words)

    # Try to trim at the last complete sentence boundary if feasible
    for p in [".", "!", "?"]:
        last_idx = result.rfind(p)
        if last_idx > 0 and len(result[: last_idx + 1].split()) >= 5:
            return result[: last_idx + 1]

    if not result.endswith((".", "!", "?")):
        result += "."
    return result


_NEW_PROMPT = """\
You are a warm, friendly host at an openhouse event.
Reference Document:
{doc_info}

A new visitor has just arrived. Generate a short, personal welcome message.
Use their name naturally. Keep it strictly under 25 words (maximum 30 words). Do not add any preamble or explanation — just the message to speak aloud.
Visitor name: {name}
"""

_RETURN_PROMPT = """\
You are a warm, friendly host at an openhouse event.
Reference Document:
{doc_info}

A returning visitor has come back. Generate a short, personal welcome-back message.
Make it feel genuinely warm and recognisable — like you remember them.
Keep it strictly under 25 words (maximum 30 words). Do not add any preamble or explanation — just the message to speak aloud.
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
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        return ""
    except requests.exceptions.ConnectionError:
        print(f"[llm] Local LLM server not reachable at {config.LLM_BASE_URL}. Ensure LM Studio (or your local LLM server) is running.")
        return ""
    except Exception as e:
        print(f"[llm] Error calling LLM: {e}")
        return ""


def generate_chat_response(prompt: str) -> str:
    """Generate a chat response for interactive input using openhouse Markdown documentation reference."""
    doc_content = _get_openhouse_doc()
    doc_context = f"Open House Info Document:\n{doc_content}\n" if doc_content else ""

    chat_prompt = (
        "You are a warm, friendly, and knowledgeable host at an openhouse event.\n"
        f"{doc_context}"
        "Instructions: ALWAYS reference the Open House Info Document above FIRST to answer any visitor questions regarding property details, price, features, address, agent, or event info.\n"
        f"User says: {prompt}\n"
        "Keep your reply concise, warm, helpful, and strictly under 25 words (maximum 30 words). "
        "Do not add any preamble or explanation — speak the response directly."
    )
    try:
        reply = _call_llm(chat_prompt)
        if reply:
            return _trim_to_word_limit(reply, max_words=30)
    except Exception:
        pass
    return _trim_to_word_limit(
        "Welcome! Please feel free to look around or ask me anything about this property!", max_words=30
    )


def generate_new_visitor_greeting(name: str) -> str:
    doc_content = _get_openhouse_doc()
    prompt = _NEW_PROMPT.format(name=name, doc_info=doc_content or "N/A")
    try:
        reply = _call_llm(prompt)
        if reply:
            return _trim_to_word_limit(reply, max_words=30)
    except Exception:
        pass
    return _trim_to_word_limit(
        f"Hello {name}! Welcome to our open house! Feel free to look around and ask any questions.", max_words=30
    )


def generate_return_greeting(name: str, last_seen: str) -> str:
    doc_content = _get_openhouse_doc()
    prompt = _RETURN_PROMPT.format(name=name, last_seen=last_seen, doc_info=doc_content or "N/A")
    try:
        reply = _call_llm(prompt)
        if reply:
            return _trim_to_word_limit(reply, max_words=30)
    except Exception:
        pass
    return _trim_to_word_limit(
        f"Welcome back, {name}! Great to see you again at our open house!", max_words=30
    )
