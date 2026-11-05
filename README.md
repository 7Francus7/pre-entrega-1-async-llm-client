# Pre-entrega 1: Cliente de LLM robusto y asíncrono

Cliente unificado en Python 3.12 para OpenAI y Anthropic. Implementa una interfaz
común mediante `AsyncLLMManager`, llamadas asíncronas con los SDKs oficiales,
streaming de texto, validación con Pydantic y manejo controlado de errores.

## Instalación

Desde la carpeta del repositorio, crear el entorno virtual e instalar las
dependencias de `requirements.txt`.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux y macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Si `.env` ya existe, editarlo sin reemplazarlo. Contiene las credenciales
locales y está excluido de Git.

## Variables de entorno

Configurar `.env` junto a `main.py`. Las variables del sistema tienen prioridad
sobre los valores del archivo.

| Variable | Valor predeterminado | Descripción |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | Proveedor activo: `openai` o `anthropic` |
| `OPENAI_API_KEY` | Vacío | Clave requerida si el proveedor activo es OpenAI |
| `ANTHROPIC_API_KEY` | Vacío | Clave requerida si el proveedor activo es Anthropic |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Modelo de OpenAI habilitado en la cuenta |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Modelo de Anthropic habilitado en la cuenta |
| `LLM_TEMPERATURE` | `0.7` | Rango de 0 a 2 para OpenAI y de 0 a 1 para Anthropic |
| `LLM_MAX_TOKENS` | `256` | Límite de salida; entero mayor que cero |
| `LLM_TIMEOUT` | `30` | Timeout de red por operación, en segundos |
| `LLM_MAX_RETRIES` | `2` | Reintentos adicionales del SDK; entre 0 y 5 |

Solo se necesita la clave del proveedor activo. Para usar Anthropic, establecer
`LLM_PROVIDER=anthropic` y completar `ANTHROPIC_API_KEY`. Los nombres de modelo
son configurables y deben estar disponibles en la cuenta correspondiente.

## Ejecución

Windows:

```powershell
.\.venv\Scripts\python.exe main.py
```

Linux y macOS:

```bash
.venv/bin/python main.py
```

`main.py` carga `.env` y realiza la pregunta «¿Qué es la entropía?» en modo
normal y en streaming con el proveedor seleccionado. Primero imprime un
`ModelResponse` en JSON y luego los fragmentos de texto conforme llegan.
Cada ejecución realiza dos solicitudes reales y requiere credenciales válidas
y cuota disponible en el proveedor.

Los errores de red, timeout, autenticación y límite de tasa se informan de forma
controlada. Los reintentos están limitados por `LLM_MAX_RETRIES`. Un stream
interrumpido no se reinicia para evitar duplicar texto. El script devuelve
código de salida `0` si ambos modos funcionan, `1` ante errores controlados y
`130` si se cancela con Ctrl+C.

## Pruebas sin credenciales

Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux y macOS:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Las pruebas utilizan los SDKs oficiales con un transporte HTTP simulado.
Verifican selección de proveedores, validación, generación, streaming y manejo
de errores sin claves ni consumo de crédito. No comprueban el acceso a los
servicios reales.
