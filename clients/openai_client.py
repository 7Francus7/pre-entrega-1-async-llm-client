"""OpenAI Chat Completions mediante AsyncOpenAI."""

from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from clients.base import BaseLLMClient
from errors import LLMClientError
from schemas import ChatMessage, ErrorInfo, ModelConfig, ModelResponse, TokenUsage


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str, *, sdk: AsyncOpenAI | None = None):
        super().__init__(config)
        self.sdk = sdk if sdk is not None else AsyncOpenAI(
            api_key=api_key, timeout=config.timeout, max_retries=config.max_retries,
        )

    def _messages(self, messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
        # Pydantic ya validó los datos; cast solo informa al analizador de tipos.
        return cast(list[ChatCompletionMessageParam], [message.model_dump() for message in messages])

    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse:
        response = await self.sdk.chat.completions.create(
            model=self.config.model, messages=self._messages(messages),
            temperature=self.config.temperature, max_completion_tokens=self.config.max_tokens,
        )
        if not response.choices or not response.choices[0].message.content:
            raise LLMClientError(ErrorInfo(code="empty_response", message="OpenAI no devolvió texto."))
        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            provider="openai", model=response.model, content=choice.message.content,
            finish_reason=choice.finish_reason,
            usage=TokenUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens) if usage else None,
        )

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self.sdk.chat.completions.create(
            model=self.config.model, messages=self._messages(messages),
            temperature=self.config.temperature, max_completion_tokens=self.config.max_tokens,
            stream=True,
        )
        async with stream:
            async for chunk in stream:
                # Algunos eventos traen solo el rol, métricas o fin de generación.
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    async def close(self) -> None:
        await self.sdk.close()
