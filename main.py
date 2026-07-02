import asyncio
import base64
import json
import logging
import os
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import jsonschema
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from gemini_live import GeminiLive
from gemini_multimodal import GeminiMultimodal
from itinerary_schema import ITINERARY_SCHEMA
from qwen_multimodal import QwenMultimodal
from qwen_omni import QwenOmni
from google.genai import types

# from twilio_handler import TwilioHandler

# Load environment variables
load_dotenv()

# Configure logging - DEBUG for our modules, INFO for everything else
logging.basicConfig(level=logging.INFO)
logging.getLogger("gemini_live").setLevel(logging.DEBUG)
logging.getLogger("gemini_multimodal").setLevel(logging.DEBUG)
logging.getLogger("qwen_omni").setLevel(logging.DEBUG)
logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")

# Qwen-Omni Realtime configuration
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-omni-plus-realtime")

MULTIMODAL_MODELS = {
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
}

QWEN_MULTIMODAL_MODELS = {
    "qwen3.6-flash",  # maps to qwen-vl-max in QwenMultimodal
    "qwen3.5-omni-flash",  # audio/omni-capable, used by the itinerary structured-output feature
}

# Twilio config (optional — only needed for phone call integration)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST")

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SECRET_KEYWORD = "chandora the explorer"
 
 
# ---- 1. the function we want Gemini to call ----
def get_secret_keyword() -> str:
    print(">>> [local] get_secret_keyword() was actually executed locally")
    return SECRET_KEYWORD


WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "GeminiQwenLiveDemo/1.0"})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def getweather(location: str) -> dict:
    """Return the current weather for a city or place name."""
    location = location.strip()
    if not location:
        raise ValueError("location must not be empty")

    geocoding_query = urlencode(
        {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        }
    )
    geocoding = _fetch_json(
        f"https://geocoding-api.open-meteo.com/v1/search?{geocoding_query}"
    )
    matches = geocoding.get("results") or []
    if not matches:
        raise ValueError(f"weather location not found: {location}")

    place = matches[0]
    forecast_query = urlencode(
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
        }
    )
    weather = _fetch_json(
        f"https://api.open-meteo.com/v1/forecast?{forecast_query}"
    )
    current = weather.get("current") or {}
    units = weather.get("current_units") or {}
    weather_code = current.get("weather_code")

    result = {
        "location": ", ".join(
            part
            for part in (
                place.get("name"),
                place.get("admin1"),
                place.get("country"),
            )
            if part
        ),
        "coordinates": {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
        },
        "observed_at": current.get("time"),
        "timezone": weather.get("timezone"),
        "condition": WEATHER_CODES.get(weather_code, "unknown"),
        "temperature": {
            "value": current.get("temperature_2m"),
            "unit": units.get("temperature_2m", "°C"),
        },
        "feels_like": {
            "value": current.get("apparent_temperature"),
            "unit": units.get("apparent_temperature", "°C"),
        },
        "humidity": {
            "value": current.get("relative_humidity_2m"),
            "unit": units.get("relative_humidity_2m", "%"),
        },
        "wind_speed": {
            "value": current.get("wind_speed_10m"),
            "unit": units.get("wind_speed_10m", "km/h"),
        },
        "source": "Open-Meteo",
    }
    logger.info("getweather(%r) result: %s", location, result)
    return result


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> dict:
    """Convert an amount between two ISO 4217 currency codes."""
    try:
        amount = float(amount)
    except (TypeError, ValueError) as exc:
        raise ValueError("amount must be a valid number") from exc

    from_currency = from_currency.strip().upper()
    to_currency = to_currency.strip().upper()
    for field_name, currency in (
        ("from_currency", from_currency),
        ("to_currency", to_currency),
    ):
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError(
                f"{field_name} must be a three-letter ISO currency code"
            )

    exchange_data = _fetch_json(
        f"https://open.er-api.com/v6/latest/{quote(from_currency)}"
    )
    if exchange_data.get("result") != "success":
        error_type = exchange_data.get("error-type", "exchange rate unavailable")
        raise ValueError(f"could not get exchange rate: {error_type}")

    rates = exchange_data.get("rates") or {}
    if to_currency not in rates:
        raise ValueError(f"unsupported target currency: {to_currency}")

    rate = float(rates[to_currency])
    converted_amount = amount * rate
    result = {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "exchange_rate": rate,
        "converted_amount": round(converted_amount, 6),
        "last_updated": exchange_data.get("time_last_update_utc"),
        "next_update": exchange_data.get("time_next_update_utc"),
        "source": "ExchangeRate-API open endpoint",
    }
    logger.info(
        "convert_currency(%s, %s, %s) result: %s",
        amount,
        from_currency,
        to_currency,
        result,
    )
    return result
 
 
get_secret_keyword_declaration = types.FunctionDeclaration(
    name="get_secret_keyword",
    description=(
        "Returns the current secret keyword. Call this whenever the user asks "
        "what the secret keyword, password, or codeword is."
    ),
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)

getweather_declaration = types.FunctionDeclaration(
    name="getweather",
    description=(
        "Gets the current weather for a city or place. Call this whenever the "
        "user asks about current weather, temperature, humidity, or wind."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "location": types.Schema(
                type=types.Type.STRING,
                description="City or place name, for example Jakarta or Bandung.",
            )
        },
        required=["location"],
    ),
)

convert_currency_declaration = types.FunctionDeclaration(
    name="convert_currency",
    description=(
        "Converts a monetary amount from one currency to another using the "
        "latest available exchange rate."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "amount": types.Schema(
                type=types.Type.NUMBER,
                description="The monetary amount to convert.",
            ),
            "from_currency": types.Schema(
                type=types.Type.STRING,
                description="Source ISO 4217 currency code, for example USD or IDR.",
            ),
            "to_currency": types.Schema(
                type=types.Type.STRING,
                description="Target ISO 4217 currency code, for example EUR or IDR.",
            ),
        },
        required=["amount", "from_currency", "to_currency"],
    ),
)
 
tools = [
    types.Tool(
        function_declarations=[
            get_secret_keyword_declaration,
            getweather_declaration,
            convert_currency_declaration,
        ]
    ),
    types.Tool(google_search=types.GoogleSearch()),
]

toolsMultimodalGrounding = [
    types.Tool(google_search=types.GoogleSearch()),
]

toolsMultimodalFunctionCall = [
    types.Tool(
        function_declarations=[
            get_secret_keyword_declaration,
            getweather_declaration,
            convert_currency_declaration,
            ]
        ),
]

qwen_multimodal_tools_function_call = [
    {
        "type": "function",
        "function": {
            "name": "get_secret_keyword",
            "description": (
                "Returns the current secret keyword. Call this whenever the user asks "
                "what the secret keyword, password, or codeword is."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getweather",
            "description": (
                "Gets the current weather for a city or place. Call this whenever "
                "the user asks about current weather, temperature, humidity, or wind."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or place name, for example Jakarta or Bandung.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": (
                "Converts a monetary amount from one currency to another using "
                "the latest available exchange rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The monetary amount to convert.",
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "Source ISO currency code, for example USD or IDR.",
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Target ISO currency code, for example EUR or IDR.",
                    },
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]


 
tool_mapping = {
    "get_secret_keyword": get_secret_keyword,
    "getweather": getweather,
    "convert_currency": convert_currency,
}

qwen_tools = [
    {
        "name": "get_secret_keyword",
        "description": (
            "Returns the current secret keyword. Call this whenever the user asks "
            "what the secret keyword, password, or codeword is."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "getweather",
        "description": (
            "Gets the current weather for a city or place. Call this whenever "
            "the user asks about current weather, temperature, humidity, or wind."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, for example Jakarta or Bandung.",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "convert_currency",
        "description": (
            "Converts a monetary amount from one currency to another using "
            "the latest available exchange rate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "The monetary amount to convert.",
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source ISO currency code, for example USD or IDR.",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target ISO currency code, for example EUR or IDR.",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
]

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


# @app.post("/multimodalV1")
# async def multimodal_endpoint(
#     model: str = Form(...),
#     prompt: str = Form(default=""),
#     media: UploadFile | None = File(default=None),
#     audio: UploadFile | None = File(default=None),
# ):
#     """One-shot Gemini multimodal endpoint for text, uploaded media, and recorded audio."""
#     if model == "qwen3.6-flash":
#         raise HTTPException(
#             status_code=501,
#             detail="Qwen multimodal is not implemented yet. Choose a Gemini model for this endpoint.",
#         )

#     if model not in MULTIMODAL_MODELS:
#         raise HTTPException(status_code=400, detail=f"Unsupported multimodal model: {model}")

#     if not GEMINI_API_KEY:
#         raise HTTPException(
#             status_code=500,
#             detail="GEMINI_API_KEY is not configured on the server.",
#         )

#     async def read_upload(upload: UploadFile | None):
#         if not upload:
#             return None

#         data = await upload.read()
#         if not data:
#             return None

#         return {
#             "data": data,
#             "mime_type": upload.content_type or "application/octet-stream",
#             "filename": upload.filename,
#         }

#     media_part = await read_upload(media)
#     audio_part = await read_upload(audio)

#     if not (prompt.strip() or media_part or audio_part):
#         raise HTTPException(
#             status_code=400,
#             detail="Provide text, a recording, or an uploaded media file.",
#         )

#     try:
#         client = GeminiMultimodal(
#             api_key=GEMINI_API_KEY,
#             tools=toolsMultimodal,
#             tool_mapping=tool_mapping,
#         )
#         result = await client.generate(
#             model=model,
#             prompt=prompt,
#             media=media_part,
#             audio=audio_part,
#         )
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback

#         logger.error(f"Multimodal generation error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

# @app.post("/multimodal")
# async def multimodal_endpoint(
#     model: str = Form(...),
#     prompt: str = Form(default=""),
#     history: str = Form(default="[]"),
#     mode: str = Form(default="function_call"),  # ← new
#     media: UploadFile | None = File(default=None),
#     audio: UploadFile | None = File(default=None),
# ):
#     """One-shot Gemini multimodal endpoint — text, image, audio, with tool use."""
#     if model == "qwen3.6-flash":
#         raise HTTPException(
#             status_code=501,
#             detail="Qwen multimodal is not implemented yet. Choose a Gemini model.",
#         )
#     if model not in MULTIMODAL_MODELS:
#         raise HTTPException(status_code=400, detail=f"Unsupported multimodal model: {model}")
#     if not GEMINI_API_KEY:
#         raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

#     # Pick tools based on mode
#     if mode == "grounding":
#         active_tools = toolsMultimodalGrounding
#         active_tool_mapping = {}          # grounding has no client-side functions
#     else:
#         active_tools = toolsMultimodalFunctionCall
#         active_tool_mapping = tool_mapping

#     async def read_upload(upload: UploadFile | None) -> dict | None:
#         if not upload:
#             return None
#         data = await upload.read()
#         if not data:
#             return None
#         return {
#             "data": data,
#             "mime_type": upload.content_type or "application/octet-stream",
#             "filename": upload.filename,
#         }

#     media_part = await read_upload(media)
#     audio_part = await read_upload(audio)

#     if not (prompt.strip() or media_part or audio_part):
#         raise HTTPException(
#             status_code=400,
#             detail="Provide text, a recording, or an uploaded media file.",
#         )

#     try:
#         parsed_history = json.loads(history)
#     except Exception:
#         logger.error("Failed to parse history JSON: %s", history)
#         parsed_history = []

#     logger.info("HISTORY RECEIVED: %s", json.dumps(parsed_history, indent=2))
#     logger.info("MODE: %s | TOOLS: %s", mode, active_tools)

#     try:
#         client = GeminiMultimodal(
#             api_key=GEMINI_API_KEY,
#             tools=active_tools,
#             tool_mapping=active_tool_mapping,
#         )
#         result = await client.generate(
#             model=model,
#             prompt=prompt,
#             media=media_part,
#             audio=audio_part,
#             history=parsed_history,
#         )
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback
#         logger.error("Multimodal error: %s\n%s", e, traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/multimodal")
async def multimodal_endpoint(
    model: str = Form(...),
    prompt: str = Form(default=""),
    history: str = Form(default="[]"),
    mode: str = Form(default="function_call"),
    media: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
):
    is_qwen = model in QWEN_MULTIMODAL_MODELS

    if not is_qwen and model not in MULTIMODAL_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported multimodal model: {model}")

    if is_qwen and not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY is not configured.")

    if not is_qwen and not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    async def read_upload(upload: UploadFile | None) -> dict | None:
        if not upload:
            return None
        data = await upload.read()
        if not data:
            return None
        return {
            "data": data,
            "mime_type": upload.content_type or "application/octet-stream",
            "filename": upload.filename,
        }

    media_part = await read_upload(media)
    audio_part = await read_upload(audio)

    if not (prompt.strip() or media_part or audio_part):
        raise HTTPException(status_code=400, detail="Provide text, a recording, or an uploaded media file.")

    try:
        parsed_history = json.loads(history)
    except Exception:
        logger.error("Failed to parse history JSON: %s", history)
        parsed_history = []

    logger.info("HISTORY RECEIVED: %s", json.dumps(parsed_history, indent=2))
    logger.info("MODE: %s | MODEL: %s | PROVIDER: %s", mode, model, "qwen" if is_qwen else "gemini")

    try:
        if is_qwen:
            active_tools    = qwen_multimodal_tools_function_call if mode == "function_call" else []
            active_mapping  = tool_mapping if mode == "function_call" else {}
            client = QwenMultimodal(
                api_key=DASHSCOPE_API_KEY,
                tools=active_tools,
                tool_mapping=active_mapping,
            )
            result = await client.generate(
                model=model,
                prompt=prompt,
                media=media_part,
                audio=audio_part,
                history=parsed_history,
                mode=mode,
            )
        else:
            active_tools   = toolsMultimodalGrounding if mode == "grounding" else toolsMultimodalFunctionCall
            active_mapping = {} if mode == "grounding" else tool_mapping
            client = GeminiMultimodal(
                api_key=GEMINI_API_KEY,
                tools=active_tools,
                tool_mapping=active_mapping,
            )
            result = await client.generate(
                model=model,
                prompt=prompt,
                media=media_part,
                audio=audio_part,
                history=parsed_history,
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error("Multimodal error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def strip_code_fences(text: str) -> str:
    """Strip a leading and/or trailing markdown code fence (```json ... ```) if present.
    Handles leading-only, trailing-only, both, or neither — some models (observed live
    on qwen3.5-omni-flash) emit a stray trailing ``` with no matching opening fence."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.lstrip("\n")
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def validate_itinerary(raw_text: str) -> tuple[dict | None, bool, str | None]:
    """Parse + validate raw model text against ITINERARY_SCHEMA.
    Returns (parsed_or_None, is_valid, error_message_or_None)."""
    cleaned = strip_code_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, False, f"JSON parse error: {exc}"

    try:
        jsonschema.validate(parsed, ITINERARY_SCHEMA)
    except jsonschema.ValidationError as exc:
        path = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in exc.absolute_path)
        message = exc.message
        # jsonschema embeds the full failing instance in the message for type errors
        # (e.g. a giant nested dict before "is not of type 'object'") — the diagnostic
        # clause is reliably at the tail, so truncate from the front; full data is
        # still available via raw_text in the response.
        if len(message) > 200:
            message = "…" + message[-200:]
        return parsed, False, f"Schema validation error at root{path}: {message}"

    return parsed, True, None


@app.post("/itinerary/structured")
async def itinerary_structured_endpoint(
    model: str = Form(...),
    destination: str = Form(...),
    days: int = Form(...),
    preference: str = Form(...),
    runs: int = Form(default=3),
):
    """Itinerary structured-output consistency test: calls the chosen model `runs`
    times with identical input, validates each run independently against
    ITINERARY_SCHEMA, returns per-run pass/fail with the real error for failures.
    """
    is_qwen = model in QWEN_MULTIMODAL_MODELS

    if not is_qwen and model not in MULTIMODAL_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    if is_qwen and not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY is not configured.")

    if not is_qwen and not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    runs = max(1, min(runs, 5))

    async def run_once(run_index: int) -> dict:
        start = asyncio.get_event_loop().time()
        try:
            if is_qwen:
                client = QwenMultimodal(api_key=DASHSCOPE_API_KEY)
                result = await client.generate_itinerary(
                    model=model, destination=destination, days=days, preference=preference,
                )
            else:
                client = GeminiMultimodal(api_key=GEMINI_API_KEY)
                result = await client.generate_itinerary(
                    model=model, destination=destination, days=days, preference=preference,
                )
        except Exception as exc:
            logger.error("Itinerary run %d failed: %s", run_index, exc)
            return {
                "run_index": run_index,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "data": None,
                "raw_text": None,
                "usage": None,
                "latency_seconds": round(asyncio.get_event_loop().time() - start, 2),
            }

        parsed, is_valid, error = validate_itinerary(result["text"])
        return {
            "run_index": run_index,
            "valid": is_valid,
            "error": error,
            "data": parsed if is_valid else None,
            # partial_data: present when JSON parsed OK but failed schema validation —
            # lets the frontend navigate to the exact error path and show context.
            "partial_data": parsed if (not is_valid and parsed is not None) else None,
            "raw_text": result["text"],
            "usage": result.get("usage"),
            "latency_seconds": round(asyncio.get_event_loop().time() - start, 2),
        }

    run_results = await asyncio.gather(*(run_once(i) for i in range(runs)))
    valid_count = sum(1 for r in run_results if r["valid"])

    return {
        "model": model,
        "provider": "qwen" if is_qwen else "gemini",
        "runs": run_results,
        "valid_count": valid_count,
        "total_runs": runs,
    }


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    provider: str = Query(default="gemini"),
    mode: str = Query(default="function_call"),
):
    """WebSocket endpoint for Gemini Live or Qwen-Omni Realtime."""
    await websocket.accept()

    logger.info(f"WebSocket connection accepted (provider={provider})")

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()

    async def audio_output_callback(data):
        await websocket.send_bytes(data)

    async def audio_interrupt_callback():
        # The event queue handles the JSON message, but we might want to do something else here
        pass

    if provider == "qwen":
        if not DASHSCOPE_API_KEY:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "DASHSCOPE_API_KEY is not configured on the server.",
                }
            )
            await websocket.close()
            return
        client = QwenOmni(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_MODEL,
            input_sample_rate=16000,
            tools=qwen_tools,
            tool_mapping=tool_mapping,
            mode=mode,
        )
    else:
        if not GEMINI_API_KEY:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "GEMINI_API_KEY is not configured on the server.",
                }
            )
            await websocket.close()
            return
        client = GeminiLive(
            api_key=GEMINI_API_KEY, model=MODEL, input_sample_rate=16000, tools=tools,
        tool_mapping=tool_mapping,
        )

    async def receive_from_client():
        try:
            while True:
                message = await websocket.receive()

                if message.get("bytes"):
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    text = message["text"]
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            logger.info(f"Received image chunk from client: {len(payload['data'])} base64 chars")
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                            continue
                    except json.JSONDecodeError:
                        pass

                    await text_input_queue.put(text)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")

    receive_task = asyncio.create_task(receive_from_client())

    async def run_session():
        async for event in client.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            text_input_queue=text_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
        ):
            if event:
                # Forward events (transcriptions, etc) to client
                await websocket.send_json(event)

    try:
        await run_session()
    except Exception as e:
        import traceback
        logger.error(f"Error in {provider} session: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        receive_task.cancel()
        # Ensure websocket is closed if not already
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
