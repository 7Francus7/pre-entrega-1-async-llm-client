"""Demostración real: pregunta sobre entropía, respuesta completa y streaming."""

import asyncio
from contextlib import aclosing
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from errors import LLMClientError
from manager import AsyncLLMManager
from schemas import ChatMessage


async def main() -> int:
    load_dotenv(Path(__file__).with_name(".env"))
    try:
        manager = AsyncLLMManager.from_env()
    except ValidationError as exc:
        print("Configuración inválida:")
        for error in exc.errors(include_input=False, include_url=False):
            print(f"- {'.'.join(map(str, error['loc'])) or 'config'}: {error['msg']}")
        return 1
    except LLMClientError as exc:
        print(exc.error.model_dump_json(indent=2))
        return 1

    messages = [
        ChatMessage(role="system", content="Respondé en español, de forma breve y clara."),
        ChatMessage(role="user", content="¿Qué es la entropía?"),
    ]
    failed = False
    async with manager:
        print(f"Proveedor: {manager.config.provider} | Modelo: {manager.config.model}")
        print("\n--- Respuesta completa ---")
        response = await manager.generate(messages)
        print(response.model_dump_json(indent=2))
        failed = not response.ok

        print("\n--- Streaming ---")
        try:
            async with aclosing(manager.stream(messages)) as stream:
                async for text in stream:
                    print(text, end="", flush=True)
            print()
        except LLMClientError as exc:
            print("\nStreaming interrumpido; el texto anterior puede estar incompleto.")
            print(exc.error.model_dump_json(indent=2))
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nEjecución cancelada.")
        raise SystemExit(130)
