"""Pruebas offline: SDKs oficiales + transporte HTTP simulado, sin claves reales."""

import asyncio
import io
import json
import os
import unittest
from contextlib import aclosing, redirect_stdout
from unittest.mock import patch

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import ValidationError

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from errors import LLMClientError
from manager import AsyncLLMManager
from main import main
from schemas import ChatMessage, ChatRequest, Credentials, ModelConfig


MESSAGES = [
    ChatMessage(role="system", content="Respondé brevemente."),
    ChatMessage(role="user", content="¿Qué es la entropía?"),
]


def completion(provider: str) -> dict:
    if provider == "openai":
        return {
            "id": "chat-test", "object": "chat.completion", "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hola mundo"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
    return {
        "id": "msg-test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "text", "text": "Hola mundo"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def sse(data: dict, event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    return (prefix + "data: " + json.dumps(data) + "\n\n").encode()


def events(provider: str) -> list[bytes]:
    if provider == "openai":
        def chunk(delta, reason=None):
            return sse({
                "id": "chat-test", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            })
        return [
            chunk({"role": "assistant"}), chunk({"content": "Hola "}),
            chunk({"content": "mundo"}), chunk({}, "stop"), b"data: [DONE]\n\n",
        ]
    message = completion("anthropic") | {"content": [], "stop_reason": None}
    return [
        sse({"type": "message_start", "message": message}, "message_start"),
        sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}, "content_block_start"),
        sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hola "}}, "content_block_delta"),
        sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "mundo"}}, "content_block_delta"),
        sse({"type": "content_block_stop", "index": 0}, "content_block_stop"),
        sse({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 3}}, "message_delta"),
        sse({"type": "message_stop"}, "message_stop"),
    ]


class EventStream(httpx.AsyncByteStream):
    def __init__(self, provider, *, gate=None, break_after_text=False, empty=False):
        self.provider = provider
        self.gate = gate
        self.break_after_text = break_after_text
        self.empty = empty
        self.closed = False

    async def __aiter__(self):
        for event in events(self.provider):
            if self.empty and (b'"text_delta"' in event or b'"content": "' in event):
                continue
            if self.gate is not None and b"mundo" in event:
                await self.gate.wait()
            yield event
            if self.break_after_text and b"Hola " in event:
                raise httpx.ReadError("Desconexión simulada")

    async def aclose(self):
        self.closed = True


class SchemaTests(unittest.TestCase):
    def test_credentials_hide_api_key(self):
        credentials = Credentials(api_key="test-secret-not-for-logs")
        self.assertNotIn("test-secret-not-for-logs", repr(credentials))
        self.assertNotIn("test-secret-not-for-logs", credentials.model_dump_json())
        self.assertEqual(credentials.api_key.get_secret_value(), "test-secret-not-for-logs")
        with self.assertRaises(ValidationError):
            Credentials(api_key="")

    def test_adapters_unwrap_secret_only_for_sdk(self):
        for provider, client_class, sdk_path in [
            ("openai", OpenAIClient, "clients.openai_client.AsyncOpenAI"),
            ("anthropic", AnthropicClient, "clients.anthropic_client.AsyncAnthropic"),
        ]:
            with self.subTest(provider=provider), patch(sdk_path) as sdk_factory:
                config = ModelConfig(provider=provider, model="test-model")
                credentials = Credentials(api_key="test-key")
                client_class(config, credentials.api_key)
                sdk_factory.assert_called_once_with(
                    api_key="test-key", timeout=config.timeout, max_retries=config.max_retries,
                )

    def test_message_validation(self):
        for data in [
            {"role": "unknown", "content": "hola"},
            {"role": "user", "content": "   "},
            {"role": "user", "content": {"nested": "no"}},
            {"role": "user", "content": "hola", "unexpected": True},
        ]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ChatMessage.model_validate(data)

    def test_config_bounds(self):
        for invalid in [
            {"temperature": -0.1}, {"temperature": 2.1}, {"temperature": float("nan")},
            {"temperature": float("inf")}, {"max_tokens": 0}, {"max_tokens": -1},
            {"max_tokens": 1.5}, {"timeout": 0}, {"max_retries": -1},
            {"max_retries": 6}, {"provider": "other"}, {"model": " "},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                ModelConfig.model_validate({"provider": "openai", "model": "test"} | invalid)
        self.assertEqual(ModelConfig(provider="openai", model="test", temperature=2).temperature, 2)
        self.assertEqual(ModelConfig(provider="anthropic", model="test", temperature=1).temperature, 1)
        with self.assertRaises(ValidationError):
            ModelConfig(provider="anthropic", model="test", temperature=1.1)

    def test_conversation_validation(self):
        for messages in [[], [MESSAGES[0]], [ChatMessage(role="assistant", content="hola")], [MESSAGES[1], MESSAGES[0]]]:
            with self.subTest(messages=messages), self.assertRaises(ValidationError):
                ChatRequest(messages=messages)

    def test_base_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseLLMClient(ModelConfig(provider="openai", model="test"))


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, provider, handler, *, retries=0):
        config = ModelConfig(provider=provider, model="test-model", max_retries=retries)
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        if provider == "openai":
            sdk = AsyncOpenAI(api_key="test-key", http_client=http_client, max_retries=retries)
            client = OpenAIClient(config, "test-key", sdk=sdk)
        else:
            sdk = AsyncAnthropic(api_key="test-key", http_client=http_client, max_retries=retries)
            client = AnthropicClient(config, "test-key", sdk=sdk)
        self.addAsyncCleanup(client.close)
        return client

    async def test_normal_response_and_request_translation(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                requests = []

                def handler(request):
                    requests.append(request)
                    return httpx.Response(200, json=completion(provider))

                response = await self.make_client(provider, handler).generate(MESSAGES)
                self.assertTrue(response.ok)
                self.assertEqual(response.content, "Hola mundo")
                self.assertEqual(response.provider, provider)
                self.assertEqual(response.usage.input_tokens, 5)
                self.assertEqual(response.usage.output_tokens, 3)
                body = json.loads(requests[0].content)
                if provider == "openai":
                    self.assertEqual(body["messages"][0]["role"], "system")
                    self.assertEqual(body["max_completion_tokens"], 256)
                    self.assertEqual(requests[0].url.path, "/v1/chat/completions")
                else:
                    self.assertEqual(body["system"], MESSAGES[0].content)
                    self.assertEqual(body["messages"], [MESSAGES[1].model_dump()])
                    self.assertEqual(body["max_tokens"], 256)
                    self.assertEqual(requests[0].url.path, "/v1/messages")

    async def test_stream_is_incremental_and_closes(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                gate = asyncio.Event()
                transport_stream = EventStream(provider, gate=gate)
                client = self.make_client(provider, lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=transport_stream,
                ))
                async with aclosing(client.stream(MESSAGES)) as stream:
                    # El segundo fragmento está bloqueado: el primero debe llegar igual.
                    self.assertEqual(await asyncio.wait_for(anext(stream), 15), "Hola ")
                    gate.set()
                    self.assertEqual([text async for text in stream], ["mundo"])
                self.assertTrue(transport_stream.closed)

    async def test_early_stream_close_releases_http(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                source = EventStream(provider)
                client = self.make_client(provider, lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=source,
                ))
                async with aclosing(client.stream(MESSAGES)) as stream:
                    await anext(stream)
                self.assertTrue(source.closed)

    async def test_errors_are_controlled_in_both_modes(self):
        for provider in ("openai", "anthropic"):
            for status, code in [(401, "authentication"), (403, "permission"), (429, "rate_limit"), (500, "api_error"), (400, "api_error")]:
                with self.subTest(provider=provider, status=status):
                    def handler(request):
                        return httpx.Response(status, json={"error": {"type": "test_error", "message": "secret-should-not-leak"}})
                    client = self.make_client(provider, handler)
                    result = await client.generate(MESSAGES)
                    self.assertFalse(result.ok)
                    self.assertEqual(result.error.code, code)
                    self.assertEqual(result.error.status_code, status)
                    self.assertNotIn("secret-should-not-leak", result.model_dump_json())
                    with self.assertRaises(LLMClientError) as caught:
                        _ = [text async for text in client.stream(MESSAGES)]
                    self.assertEqual(caught.exception.error.code, code)

    async def test_network_and_timeout_errors(self):
        for provider in ("openai", "anthropic"):
            for exception_type, code in [(httpx.ConnectError, "network"), (httpx.ReadTimeout, "timeout")]:
                with self.subTest(provider=provider, code=code):
                    def handler(request):
                        raise exception_type("Fallo simulado", request=request)
                    client = self.make_client(provider, handler)
                    self.assertEqual((await client.generate(MESSAGES)).error.code, code)
                    with self.assertRaises(LLMClientError) as caught:
                        _ = [text async for text in client.stream(MESSAGES)]
                    self.assertEqual(caught.exception.error.code, code)

    async def test_partial_stream_is_not_retried(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                calls = []
                source = EventStream(provider, break_after_text=True)

                def handler(request):
                    calls.append(request)
                    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=source)

                client = self.make_client(provider, handler, retries=2)
                received = []
                with self.assertRaises(LLMClientError) as caught:
                    async for text in client.stream(MESSAGES):
                        received.append(text)
                self.assertEqual(received, ["Hola "])
                self.assertEqual(caught.exception.error.code, "network")
                self.assertEqual(len(calls), 1)
                self.assertTrue(source.closed)

    async def test_retry_rate_limit_then_success(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                calls = []

                def handler(request):
                    calls.append(request)
                    if len(calls) == 1:
                        return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "rate limit", "type": "rate_limit_error"}})
                    return httpx.Response(200, json=completion(provider))

                result = await self.make_client(provider, handler, retries=2).generate(MESSAGES)
                self.assertTrue(result.ok)
                self.assertEqual(len(calls), 2)

    async def test_retry_exhaustion_is_bounded(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                calls = []

                def handler(request):
                    calls.append(request)
                    return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "rate limit", "type": "rate_limit_error"}})

                result = await self.make_client(provider, handler, retries=2).generate(MESSAGES)
                self.assertEqual(result.error.code, "rate_limit")
                self.assertEqual(len(calls), 3)

    async def test_invalid_input_never_calls_network(self):
        for provider in ("openai", "anthropic"):
            def handler(request):
                self.fail("Una entrada inválida no debe llamar a la API.")
            client = self.make_client(provider, handler)
            self.assertEqual((await client.generate([])).error.code, "invalid_input")
            with self.assertRaises(LLMClientError) as caught:
                _ = [text async for text in client.stream([])]
            self.assertEqual(caught.exception.error.code, "invalid_input")

    async def test_empty_normal_and_stream_responses(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                def handler(request):
                    if json.loads(request.content).get("stream"):
                        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=EventStream(provider, empty=True))
                    data = completion(provider)
                    if provider == "openai":
                        data["choices"][0]["message"]["content"] = None
                    else:
                        data["content"] = []
                    return httpx.Response(200, json=data)
                client = self.make_client(provider, handler)
                self.assertEqual((await client.generate(MESSAGES)).error.code, "empty_response")
                with self.assertRaises(LLMClientError) as caught:
                    _ = [text async for text in client.stream(MESSAGES)]
                self.assertEqual(caught.exception.error.code, "empty_response")

    async def test_calls_do_not_block_event_loop(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                entered = asyncio.Event()
                release = asyncio.Event()

                async def handler(request):
                    entered.set()
                    await release.wait()
                    return httpx.Response(200, json=completion(provider))

                client = self.make_client(provider, handler)
                task = asyncio.create_task(client.generate(MESSAGES))
                try:
                    # El plazo es solo una guarda contra cuelgues, no una medición
                    # de rendimiento: el SDK puede cargar módulos al primer uso.
                    await asyncio.wait_for(entered.wait(), 15)
                    self.assertFalse(task.done())
                    # Esta corrutina sigue ejecutándose mientras la petición espera.
                    release.set()
                    self.assertTrue((await asyncio.wait_for(task, 15)).ok)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_cancellation_is_not_swallowed(self):
        entered = asyncio.Event()

        async def handler(request):
            entered.set()
            await asyncio.Event().wait()

        client = self.make_client("openai", handler)
        task = asyncio.create_task(client.generate(MESSAGES))
        await asyncio.wait_for(entered.wait(), 15)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


class ManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_success_and_controlled_failure(self):
        for provider in ("openai", "anthropic"):
            for status in (200, 401, 429):
                with self.subTest(provider=provider, status=status):
                    requests = []

                    def handler(request):
                        requests.append(request)
                        if status != 200:
                            return httpx.Response(status, json={"error": {"type": "test_error", "message": "test"}})
                        if json.loads(request.content).get("stream"):
                            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=EventStream(provider))
                        return httpx.Response(200, json=completion(provider))

                    config = ModelConfig(provider=provider, model="test-model", max_retries=0)
                    helper = ClientTests()
                    client = helper.make_client(provider, handler)
                    class_name = "OpenAIClient" if provider == "openai" else "AnthropicClient"
                    with patch(f"manager.{class_name}", return_value=client):
                        manager = AsyncLLMManager(config, "test-key")
                    output = io.StringIO()
                    with patch("main.load_dotenv"), patch("main.AsyncLLMManager.from_env", return_value=manager), redirect_stdout(output):
                        self.assertEqual(await main(), 0 if status == 200 else 1)
                    self.assertEqual(len(requests), 2)
                    self.assertIn("Streaming", output.getvalue())
                    self.assertNotIn("Traceback", output.getvalue())
                    self.assertTrue(client.sdk.is_closed())

    async def test_provider_selection_and_env_config(self):
        for provider, expected in [("openai", OpenAIClient), ("anthropic", AnthropicClient)]:
            with self.subTest(provider=provider), patch.dict(os.environ, {
                "LLM_PROVIDER": provider.upper(), f"{provider.upper()}_API_KEY": "test-key",
                f"{provider.upper()}_MODEL": "test-model", "LLM_MAX_TOKENS": "128",
                "LLM_TEMPERATURE": "0.5", "LLM_TIMEOUT": "5", "LLM_MAX_RETRIES": "0",
            }, clear=True):
                async with AsyncLLMManager.from_env() as manager:
                    self.assertIsInstance(manager.client, expected)
                    self.assertEqual(manager.config.max_tokens, 128)
                    self.assertEqual(manager.config.timeout, 5)
                    self.assertEqual(manager.config.temperature, 0.5)
                    sdk = manager.client.sdk
                    self.assertFalse(sdk.is_closed())
                self.assertTrue(sdk.is_closed())

    async def test_manager_delegates_generation_and_stream(self):
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                def handler(request):
                    if json.loads(request.content).get("stream"):
                        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=EventStream(provider))
                    return httpx.Response(200, json=completion(provider))

                helper = ClientTests()
                client = helper.make_client(provider, handler)
                config = ModelConfig(provider=provider, model="test-model")
                class_name = "OpenAIClient" if provider == "openai" else "AnthropicClient"
                with patch(f"manager.{class_name}", return_value=client):
                    async with AsyncLLMManager(config, "test-key") as manager:
                        self.assertTrue((await manager.generate(MESSAGES)).ok)
                        self.assertEqual("".join([t async for t in manager.stream(MESSAGES)]), "Hola mundo")

    async def test_missing_key_and_bad_env(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(LLMClientError) as caught:
            AsyncLLMManager.from_env()
        self.assertEqual(caught.exception.error.code, "configuration")
        for invalid in [{"LLM_PROVIDER": "unknown"}, {"LLM_MAX_TOKENS": "abc"}]:
            with patch.dict(os.environ, invalid, clear=True), self.assertRaises(ValidationError):
                AsyncLLMManager.from_env()


if __name__ == "__main__":
    unittest.main()
