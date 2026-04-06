"""
Gemini API client for RLM sub-LLM calls.

Provides llm_completion() and llm_completion_batch() for the REPL environment.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai

_client: genai.Client | None = None
_client_lock = threading.Lock()

CALL_TIMEOUT = 120


def get_client() -> genai.Client:
    """Thread-safe lazy Gemini client initialization."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            _client = genai.Client(api_key=api_key)
    return _client


def llm_completion(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Single Gemini completion. Returns the response text."""
    client = get_client()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
        )
    except Exception as e:
        return f"[LLM_ERROR] {e}"
    if response.text is None:
        candidates = getattr(response, "candidates", [])
        if candidates:
            reason = getattr(candidates[0], "finish_reason", "unknown")
            return f"[LLM_ERROR] Empty response (finish_reason: {reason})"
        return "[LLM_ERROR] Empty response from Gemini"
    return response.text


def llm_completion_batch(
    prompts: list[str],
    model: str = "gemini-2.5-flash",
    max_workers: int = 8,
) -> list[str]:
    """
    Parallel Gemini completions via ThreadPoolExecutor.

    Returns results in the same order as the input prompts.
    Failed calls return the error message as a string.
    """
    results: list[str | None] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(llm_completion, prompt, model): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx, timeout=CALL_TIMEOUT):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result(timeout=CALL_TIMEOUT)
            except TimeoutError:
                results[idx] = "[LLM_ERROR] Gemini call timed out"
            except Exception as e:
                results[idx] = f"[LLM_ERROR] {e}"

    return [r if r is not None else "[LLM_ERROR] No result" for r in results]
