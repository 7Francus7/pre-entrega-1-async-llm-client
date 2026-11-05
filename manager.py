"""Punto de entrada común: selecciona un proveedor por configuración."""

import os
from collections.abc import AsyncIterator
from typing import Self

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from errors import LLMClientError
from schemas import ChatMessage, ErrorInfo, ModelConfig, ModelResponse


class AsyncLLMManager:
    def __init__(self, config: ModelConfig, api_key: str):
        if not api_key.strip():
            raise LLMClientError(ErrorInfo(
                code="configuration", message=f"Falta {config.provider.upper()}_API_KEY en el entorno o .env.",
            ))
        self.config = config
        client_class = OpenAIClient if config.provider == "openai" else AnthropicClient
        self.client: BaseLLMClient = client_class(config, api_key.strip())

    @classmethod
    def from_env(cls) -> Self:
        # main.py carga .env; la biblioteca también acepta variables del sistema.
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        default_model = "gpt-4.1-mini" if provider == "openai" else "claude-haiku-4-5"
        config = ModelConfig.model_validate({
            "provider": provider,
            "model": os.getenv(f"{provider.upper()}_MODEL", default_model),
            "temperature": os.getenv("LLM_TEMPERATURE", "0.7"),
            "max_tokens": os.getenv("LLM_MAX_TOKENS", "256"),
            "timeout": os.getenv("LLM_TIMEOUT", "30"),
            "max_retries": os.getenv("LLM_MAX_RETRIES", "2"),
        })
        return cls(config, os.getenv(f"{provider.upper()}_API_KEY", ""))

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        return self.client.stream(messages)

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
