"""Thin wrapper around a llama.cpp OpenAI-compatible local server.

llama.cpp sends thinking tokens in delta.reasoning_content and prose
tokens in delta.content — separate fields, no tag parsing needed.

StreamChunk.is_thinking=True  → came from delta.reasoning_content
StreamChunk.is_thinking=False → came from delta.content

For non-streaming calls, the same split appears on message.reasoning_content
and message.content.

Import contract
~~~~~~~~~~~~~~~
llm_interface.py has no imports from other engine modules.
inference.py imports LLMClient, LLMResponse, StreamChunk from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Result from a completed (non-streaming) LLM call.

    Args:
        prose: Text from message.content. What the story uses.
        thinking: Text from message.reasoning_content. The model's
            reasoning before writing. None if the model produced none.
        tokens_used: Total tokens reported by the server, or None.
    """

    prose: str
    thinking: str | None
    tokens_used: int | None


@dataclass
class StreamChunk:
    """One token from a streaming generation call.

    Args:
        token: The token text.
        is_thinking: True = from reasoning_content (model's thinking).
                     False = from content (prose for the story).
    """

    token: str
    is_thinking: bool


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Wrapper around a llama.cpp OpenAI-compatible server.

    max_tokens must cover both thinking and prose. With an 8096-token
    context and ~1600-token prompts, max_tokens=3000 works well:
      thinking: typically 500–1500 tokens
      prose:    typically 600–1500 tokens

    Args:
        base_url: Server URL. Default: http://127.0.0.1:8080/v1
        api_key: Ignored by llama.cpp.
        model: Ignored by llama.cpp.
        timeout: Request timeout in seconds. Thinking models are slow —
            300 seconds (5 min) is a safe ceiling for long scenes.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "local",
        model: str = "local",
        timeout: float = 300.0,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._model = model

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 3000,
        temperature: float = 0.88,
        top_p: float = 0.95,
    ) -> LLMResponse:
        """Blocking generation. Returns complete LLMResponse.

        thinking comes from message.reasoning_content.
        prose comes from message.content.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        message = response.choices[0].message
        prose = (message.content or "").strip()
        thinking_raw = getattr(message, "reasoning_content", None)
        thinking = thinking_raw.strip() if thinking_raw else None
        tokens = response.usage.total_tokens if response.usage else None

        return LLMResponse(prose=prose, thinking=thinking, tokens_used=tokens)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 3000,
        temperature: float = 0.88,
        top_p: float = 0.95,
    ) -> Iterator[StreamChunk]:
        """Streaming generation. Yields StreamChunk per token.

        Thinking tokens (reasoning_content) → is_thinking=True
        Prose tokens (content)              → is_thinking=False

        Both are yielded in arrival order. The caller decides how to
        display or accumulate them.
        """
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta

            # Thinking tokens come in via reasoning_content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield StreamChunk(token=reasoning, is_thinking=True)

            # Prose tokens come in via content
            content = getattr(delta, "content", None)
            if content:
                yield StreamChunk(token=content, is_thinking=False)