"""La base comparte validación y errores; cada proveedor adapta su protocolo."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import aclosing

from pydantic import ValidationError

from errors import API_ERRORS, LLMClientError, normalize_error
from schemas import ChatMessage, ChatRequest, ErrorInfo, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Respuesta completa o error estructurado; los errores de programación se propagan."""
        try:
            request = ChatRequest(messages=messages)
            return await self._generate(request.messages)
        except ValidationError:
            error = ErrorInfo(code="invalid_input", message="Mensajes inválidos: revisá roles, contenido y orden.")
        except LLMClientError as exc:
            error = exc.error
        except API_ERRORS as exc:
            error = normalize_error(exc)
        return ModelResponse(provider=self.config.provider, model=self.config.model, error=error)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Entrega texto incremental. Un fallo se informa como LLMClientError, nunca como texto."""
        try:
            request = ChatRequest(messages=messages)
            received_text = False
            # aclosing libera el stream interno incluso si el consumidor lo cancela.
            async with aclosing(self._stream(request.messages)) as chunks:
                async for chunk in chunks:
                    if chunk:
                        received_text = True
                        yield chunk
            if not received_text:
                raise LLMClientError(ErrorInfo(code="empty_response", message="El modelo no devolvió texto."))
        except ValidationError as exc:
            raise LLMClientError(ErrorInfo(
                code="invalid_input", message="Mensajes inválidos: revisá roles, contenido y orden.",
            )) from exc
        except API_ERRORS as exc:
            # No reiniciamos un stream parcial: repetirlo podría duplicar texto.
            raise LLMClientError(normalize_error(exc)) from exc

    @abstractmethod
    async def _generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Operación de generación específica del proveedor."""

    @abstractmethod
    def _stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Implementado con async def y yield en cada proveedor."""

    @abstractmethod
    async def close(self) -> None:
        """Cierra las conexiones HTTP del SDK."""
