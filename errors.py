"""Errores uniformes. Nunca exponemos el cuerpo crudo de una API ni sus claves."""

import anthropic
import httpx
import openai

from schemas import ErrorInfo

API_ERRORS = (openai.APIError, anthropic.APIError, httpx.HTTPError)


class LLMClientError(Exception):
    """Error controlado para configuración y generadores asíncronos."""

    def __init__(self, error: ErrorInfo):
        self.error = error
        super().__init__(error.message)


def normalize_error(exc: Exception) -> ErrorInfo:
    status = getattr(exc, "status_code", None)
    if isinstance(exc, (openai.APITimeoutError, anthropic.APITimeoutError, httpx.TimeoutException)):
        return ErrorInfo(code="timeout", message="Se agotó el tiempo de espera.", retryable=True)
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError, httpx.TransportError)):
        return ErrorInfo(code="network", message="No se pudo conectar con el proveedor.", retryable=True)
    if status == 401:
        return ErrorInfo(code="authentication", message="API key inválida o revocada.", status_code=status)
    if status == 403:
        return ErrorInfo(code="permission", message="La cuenta no tiene permiso para esta operación.", status_code=status)
    if status == 429:
        return ErrorInfo(
            code="rate_limit", message="Límite de solicitudes o cuota alcanzado. Revisá cuota y facturación.",
            retryable=True, status_code=status,
        )
    return ErrorInfo(
        code="api_error", message="El proveedor rechazó la solicitud o interrumpió la respuesta.",
        retryable=status in (408, 409) or (status is not None and status >= 500),
        status_code=status,
    )
