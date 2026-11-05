# Pre-entrega 1: Unified Async LLM Client

Cliente unificado en **Python 3.12** para **OpenAI y Anthropic**, con generación
asíncrona, streaming de texto, validación con Pydantic y errores controlados.
El proveedor se elige mediante una variable de entorno; el código que consume
el gestor es el mismo para ambos.

## Inicio rápido en Windows (PowerShell)

Desde la carpeta del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Editá `.env` y completá la clave del proveedor elegido. No hace falta activar
el entorno virtual: usar su ejecutable evita problemas con la política de
ejecución de PowerShell. Si `.env` ya existe, conservá tus valores.

```powershell
.\.venv\Scripts\python.exe main.py
```

En Linux o macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python main.py
```

`main.py` pregunta **«¿Qué es la entropía?»** dos veces: primero muestra una
respuesta completa en JSON; después imprime los fragmentos conforme llegan.
Son **dos solicitudes reales** que pueden consumir crédito de API.
Las pruebas offline de abajo no necesitan claves ni consumen crédito.

## Configuración

| Variable | Valor inicial | Uso |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Vacío | Obligatoria si se elige OpenAI |
| `ANTHROPIC_API_KEY` | Vacío | Obligatoria si se elige Anthropic |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Modelo de OpenAI habilitado en tu cuenta |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Modelo de Anthropic habilitado en tu cuenta |
| `LLM_TEMPERATURE` | `0.7` | OpenAI: de 0 a 2; Anthropic: de 0 a 1 |
| `LLM_MAX_TOKENS` | `256` | Entero mayor que cero; límite de salida |
| `LLM_TIMEOUT` | `30` | Timeout de red por operación, en segundos |
| `LLM_MAX_RETRIES` | `2` | Reintentos adicionales del SDK, entre 0 y 5 |

Para cambiar de proveedor, modificá `LLM_PROVIDER=anthropic` y completá
`ANTHROPIC_API_KEY`. No se necesita una clave del proveedor inactivo.
Los modelos se configuran por separado para evitar enviar el nombre de un
modelo de OpenAI a Anthropic o viceversa.

El archivo `.env` se busca junto a `main.py`. Las variables ya definidas en el
sistema tienen prioridad sobre `.env`, por el comportamiento de `load_dotenv`.
`.env` está excluido de Git; `.env.example` solo contiene campos vacíos.

El gestor valida las credenciales mediante `Credentials`, cuyo campo
`api_key` es `SecretStr` de Pydantic. Su representación y serialización JSON
ocultan la clave. El adaptador usa `get_secret_value()` al construir el SDK.
Esto evita exposiciones accidentales al mostrar el esquema; no cifra la clave
ni reemplaza el cuidado del archivo `.env` o la configuración de los logs HTTP.

### Compatibilidad

Se fijan `openai==2.8.1`, `anthropic==0.74.1` y `httpx==0.28.1` para conservar
una interfaz compatible con la actividad, incluido `temperature`. Todas las
dependencias resueltas están fijadas en `requirements.txt`; `requirements.in`
declara las dependencias directas. No actualizar los SDKs sin revisar sus
cambios y ejecutar las pruebas: versiones posteriores pueden modificar los
parámetros admitidos.

Los nombres de modelo son ejemplos configurables. Su disponibilidad y los
parámetros admitidos dependen del proveedor y de la cuenta. Pydantic valida los
rangos comunes y la restricción adicional de Anthropic; un modelo con reglas
más específicas puede rechazar una solicitud con un error controlado.
El timeout se aplica a operaciones de red; los reintentos pueden extender el
tiempo total de la solicitud. No se implementa un plazo total de generación.

## Estructura

```text
.
├── schemas.py                  # Mensajes, configuración, credenciales y respuesta
├── errors.py                   # Excepción común y traducción de errores
├── manager.py                  # AsyncLLMManager y selección desde el entorno
├── clients/
│   ├── base.py                 # BaseLLMClient abstracto: contrato y lógica común
│   ├── openai_client.py        # Adaptador de AsyncOpenAI
│   └── anthropic_client.py     # Adaptador de AsyncAnthropic
├── main.py                     # Demostración completa y streaming
├── tests/test_llm.py            # Pruebas offline con los SDKs reales
├── .github/workflows/tests.yml # Pruebas automáticas en Python 3.12
├── .env.example
├── requirements.in
└── requirements.txt
```

## Interfaz común

```python
import asyncio
from contextlib import aclosing

from dotenv import load_dotenv

from errors import LLMClientError
from manager import AsyncLLMManager
from schemas import ChatMessage


async def ejemplo():
    load_dotenv()
    mensajes = [ChatMessage(role="user", content="¿Qué es la entropía?")]

    async with AsyncLLMManager.from_env() as manager:
        respuesta = await manager.generate(mensajes)
        if respuesta.ok:
            print(respuesta.content)
        else:
            print(respuesta.error.model_dump())

        try:
            async with aclosing(manager.stream(mensajes)) as stream:
                async for fragmento in stream:
                    print(fragmento, end="", flush=True)
        except LLMClientError as error:
            print(error.error.model_dump())


asyncio.run(ejemplo())
```

También se puede construir `ModelConfig` y pasarlo a
`AsyncLLMManager(config, api_key)` sin usar variables de entorno.

`generate()` devuelve `ModelResponse`, con proveedor, modelo, contenido,
motivo de finalización, uso de tokens y un posible error. El uso de tokens
puede ser `None` si el proveedor no lo informa.

`stream()` devuelve un generador asíncrono de `str`. Cada elemento es un
fragmento de texto del proveedor: puede contener parte de una palabra o varios
tokens. No existe garantía de «un fragmento = un token». Los eventos de
metadatos se filtran; el texto se entrega inmediatamente, sin acumular toda
la respuesta ni simular streaming dividiendo una respuesta completa.

`aclosing` asegura la liberación inmediata del generador si el consumidor
interrumpe el recorrido con `break`, una excepción o una cancelación.
El contexto `async with manager` cierra las conexiones HTTP al terminar.

## Resiliencia y errores

Los SDKs oficiales realizan reintentos asíncronos limitados para fallos de
conexión, timeouts, HTTP 408/409, 429 y errores 5xx. Se usa su política de
espera y `Retry-After`; no se agrega un segundo sistema de reintentos.
Con `LLM_MAX_RETRIES=2` hay como máximo tres intentos de la solicitud inicial.
Una clave inválida (401) o un permiso denegado (403) no se reintentan.

En generación normal, los fallos esperados producen `ModelResponse.error`:

```json
{
  "code": "rate_limit",
  "message": "Límite de solicitudes o cuota alcanzado. Revisá cuota y facturación.",
  "retryable": true,
  "status_code": 429
}
```

`retryable` indica una categoría potencialmente transitoria; no garantiza que
un nuevo intento funcione. Una cuota agotada requiere revisar facturación.

En streaming, los fallos se traducen a `LLMClientError`, que contiene el mismo
`ErrorInfo`. El consumidor debe capturarlo como hace `main.py`. No se mezcla
un mensaje de error con los fragmentos de respuesta. Si falla a mitad de la
transmisión, se conserva lo recibido y se avisa que está incompleto; no se
reinicia el stream, para evitar duplicar texto y llamadas.

Además se detectan entradas inválidas y respuestas sin texto. Los errores
controlados no exponen el cuerpo crudo de la API ni claves. Los errores de
programación inesperados se propagan para poder corregirlos; no se ocultan
con un `except Exception` general. `asyncio.CancelledError` se respeta.

El script captura los errores esperados y termina sin traceback. Devuelve
código de salida `0` si ambas demostraciones funcionan, `1` ante errores
controlados y `130` ante una cancelación con Ctrl+C.

## Pruebas sin API keys

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

En Linux/macOS: `.venv/bin/python -m unittest discover -s tests -v`.
Se usa `unittest` de la biblioteca estándar y `httpx.MockTransport`. Los
SDKs oficiales construyen y procesan las solicitudes reales, pero el transporte
devuelve JSON o eventos SSE locales. No hay solicitudes a los proveedores.

Las pruebas verifican ambos proveedores, selección por entorno, traducción
de `system`, respuesta normal, streaming incremental, cierre anticipado,
validación, red, timeouts, autenticación, permisos, protección de credenciales, rate limiting, reintentos
limitados, respuestas vacías, cancelación y recuperación del script principal.
También verifican que una corrutina pueda progresar mientras la llamada espera.

**Alcance de la verificación:** estas pruebas comprueban integración con los
SDKs y comportamiento local. No validan una clave real, crédito disponible,
acceso a modelos ni conectividad con los servicios. Para verificar eso, ejecutá
`main.py` con cada proveedor usando tus propias credenciales.

## Correspondencia con la rúbrica

| Criterio | Peso | Evidencia |
| --- | --- | --- |
| Abstracción y gestión de proveedores | 35 % | `AsyncLLMManager`, `BaseLLMClient` y dos adaptadores intercambiables |
| Streaming asíncrono | 25 % | Métodos con `async for` y `yield`; prueba de entrega incremental |
| Validación con Pydantic | 20 % | `ChatMessage`, `ChatRequest`, `ModelConfig`, límites y roles |
| Resiliencia y manejo de errores | 20 % | Reintentos del SDK, errores uniformes, timeouts y pruebas de fallos |

## Guía para entender y defender el código

### 1. Qué significa «unificado»

OpenAI y Anthropic tienen APIs distintas. El resto del programa usa siempre
`manager.generate()` o `manager.stream()`. El gestor selecciona un adaptador;
cada adaptador transforma el formato común al formato de su proveedor.
Esto permite cambiar de proveedor sin cambiar el código del consumidor.

### 2. Por qué usar Pydantic primero

Un diccionario permite escribir un rol inexistente, contenido anidado por error
o parámetros fuera de rango. Pydantic convierte esa entrada en un contrato
explícito y falla antes de gastar una solicitud. Se rechazan campos extra,
roles desconocidos, contenido vacío, conversaciones vacías y parámetros
inválidos. Los mensajes `system` van al principio y la conversación empieza
por `user`, como contrato común de este cliente.

`model_dump()` convierte un objeto ya validado al diccionario que espera el
SDK. `model_dump_json()` permite imprimir una respuesta estructurada.

### 3. Qué hacen `async`, `await` y el event loop

`async def` define una corrutina. Al esperar una operación de red con `await`,
la corrutina cede el control para que otras tareas puedan avanzar. El event
loop coordina esas tareas. La red no responde más rápido por usar `async`,
pero el programa puede atender otras tareas mientras espera.

Usar un SDK síncrono dentro de `async def` seguiría bloqueando. Por eso se
usan `AsyncOpenAI` y `AsyncAnthropic` y se esperan sus operaciones. La demo
ejecuta los dos modos en secuencia para poder comparar sus salidas; no necesita
concurrencia para ser asíncrona. Las pruebas verifican la espera no bloqueante.

### 4. Diferencia entre `return` y `yield`

`generate()` usa `return` para entregar una respuesta completa.
Un método `async def` con `yield` crea un generador asíncrono: cada `yield`
entrega un fragmento y permite continuar después. Se consume con `async for`,
sin hacer `await manager.stream(...)`. El método del gestor solo delega;
los generadores están implementados en la base y los adaptadores.

### 5. Qué cambia entre proveedores

OpenAI recibe los roles dentro de `messages` y su contenido incremental llega
en `chunk.choices[0].delta.content`. El límite común `max_tokens` se traduce
a `max_completion_tokens` en Chat Completions.

Anthropic recibe las instrucciones `system` en un parámetro separado. El
adaptador las extrae y envía los mensajes de conversación por separado. Su
SDK expone `stream.text_stream`; el límite se llama `max_tokens`.

### 6. Ejercicios breves

1. Cambiá la pregunta de `main.py` y explicá dónde se valida.
2. Poné `LLM_TEMPERATURE=3`: identificá por qué falla antes de llamar a la API.
3. Cambiá `LLM_PROVIDER` y explicá por qué el bucle del consumidor sigue igual.
4. Leé la prueba de desconexión parcial: explicá por qué no reintentamos.
5. Explicá qué ocurriría si reemplazaras `AsyncOpenAI` por `OpenAI`.

## Entrega

El entregable es el enlace al repositorio accesible para el docente, con este
código y README. No se necesita un informe o PDF separado. Antes de entregarlo,
ejecutá las pruebas y, cuando tengas credenciales y crédito, la demo real con
OpenAI y Anthropic. No subas `.env` ni el entorno virtual.

## Referencias oficiales

- [OpenAI Python SDK, versión utilizada](https://github.com/openai/openai-python/tree/v2.8.1)
- [Anthropic Python SDK, versión utilizada](https://github.com/anthropics/anthropic-sdk-python/tree/v0.74.1)
- [Pydantic: modelos](https://docs.pydantic.dev/latest/concepts/models/)
- [Python 3.12: asyncio](https://docs.python.org/3.12/library/asyncio.html)
- [HTTPX: transportes y simulación de HTTP](https://www.python-httpx.org/advanced/transports/)
