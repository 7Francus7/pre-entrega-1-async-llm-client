"""Contratos comunes: se validan antes de enviar datos a cualquier API."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

Provider = Literal["openai", "anthropic"]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ChatMessage(Schema):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class Credentials(Schema):
    api_key: SecretStr = Field(min_length=1)


class ModelConfig(Schema):
    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=256, gt=0)
    timeout: float = Field(default=30, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def check_provider_temperature(self) -> Self:
        if self.provider == "anthropic" and self.temperature > 1:
            raise ValueError("Anthropic admite temperatura entre 0 y 1.")
        return self


class ChatRequest(Schema):
    messages: list[ChatMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def check_message_order(self) -> Self:
        conversation_started = False
        for message in self.messages:
            if message.role == "system":
                if conversation_started:
                    raise ValueError("Los mensajes system deben ir al principio.")
            elif not conversation_started:
                if message.role != "user":
                    raise ValueError("La conversación debe comenzar con un mensaje user.")
                conversation_started = True
        if not conversation_started:
            raise ValueError("Debe existir al menos un mensaje user.")
        return self


class TokenUsage(Schema):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ErrorInfo(Schema):
    code: Literal[
        "configuration", "invalid_input", "authentication", "permission",
        "rate_limit", "timeout", "network", "api_error", "empty_response",
    ]
    message: str
    retryable: bool = False
    status_code: int | None = None


class ModelResponse(Schema):
    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    error: ErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
