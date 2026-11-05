"""Isolated async OpenAI chat client."""

import logging

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM cannot produce an answer. Message is safe for logs only."""


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APITimeoutError as exc:
            raise LLMError("OpenAI request timed out") from exc
        except RateLimitError as exc:
            raise LLMError("OpenAI rate limit exceeded") from exc
        except APIConnectionError as exc:
            raise LLMError("Could not connect to OpenAI") from exc
        except APIStatusError as exc:
            raise LLMError(f"OpenAI returned HTTP {exc.status_code}") from exc
        except APIError as exc:
            raise LLMError("OpenAI API error") from exc

        if not response.choices:
            raise LLMError("OpenAI returned no choices")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMError("OpenAI returned an empty answer")
        return content.strip()

    async def close(self) -> None:
        await self._client.close()
