"""Anthropic Messages mediante AsyncAnthropic."""

from collections.abc import AsyncIterator
from typing import cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from pydantic import SecretStr

from clients.base import BaseLLMClient
from errors import LLMClientError
from schemas import ChatMessage, ErrorInfo, ModelConfig, ModelResponse, TokenUsage


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str | SecretStr, *, sdk: AsyncAnthropic | None = None):
        super().__init__(config)
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self.sdk = sdk if sdk is not None else AsyncAnthropic(
            api_key=secret.get_secret_value(), timeout=config.timeout, max_retries=config.max_retries,
        )

    def _split_messages(self, messages: list[ChatMessage]) -> tuple[str, list[MessageParam]]:
        system = "\n\n".join(message.content for message in messages if message.role == "system")
        conversation = cast(list[MessageParam], [
            message.model_dump() for message in messages if message.role != "system"
        ])
        return system, conversation

    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse:
        system, conversation = self._split_messages(messages)
        response = await self.sdk.messages.create(
            model=self.config.model, messages=conversation, system=system,
            temperature=self.config.temperature, max_tokens=self.config.max_tokens,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise LLMClientError(ErrorInfo(code="empty_response", message="Anthropic no devolvió texto."))
        return ModelResponse(
            provider="anthropic", model=response.model, content=text, finish_reason=response.stop_reason,
            usage=TokenUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens),
        )

    async def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        system, conversation = self._split_messages(messages)
        async with self.sdk.messages.stream(
            model=self.config.model, messages=conversation, system=system,
            temperature=self.config.temperature, max_tokens=self.config.max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def close(self) -> None:
        await self.sdk.close()
