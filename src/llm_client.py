"""
Thin abstraction over the LLM backend so classify_jd.py, rewrite.py, and
verify.py don't need to care whether calls go to the paid Anthropic API or a
free local model via Ollama.

Set LLM_BACKEND=ollama (default, free, local) or LLM_BACKEND=anthropic (paid,
needs ANTHROPIC_API_KEY) as an environment variable to switch.

Ollama setup (one-time, free):
    1. Install: https://ollama.com/download
    2. Pull a model, e.g.:  ollama pull llama3.1:8b
    3. Ollama runs a local server automatically on http://localhost:11434
    No API key, no per-token cost -- just local compute.
"""
import os
import requests

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "anthropic"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    """Send a single-turn prompt to whichever backend is configured, return raw text."""
    if LLM_BACKEND == "anthropic":
        return _call_anthropic(prompt, max_tokens)
    return _call_ollama(prompt, max_tokens)


def _call_ollama(prompt: str, max_tokens: int) -> str:
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _call_anthropic(prompt: str, max_tokens: int) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


if __name__ == "__main__":
    print(f"Backend: {LLM_BACKEND}")
    print(call_llm("Say hello in exactly five words."))
