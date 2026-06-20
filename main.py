import asyncio
import base64
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from gemini_live import GeminiLive
from qwen_omni import QwenOmni
from google.genai import types

# from twilio_handler import TwilioHandler

# Load environment variables
load_dotenv()

# Configure logging - DEBUG for our modules, INFO for everything else
logging.basicConfig(level=logging.INFO)
logging.getLogger("gemini_live").setLevel(logging.DEBUG)
logging.getLogger("qwen_omni").setLevel(logging.DEBUG)
logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")

# Qwen-Omni Realtime configuration
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-omni-plus-realtime")

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
 
 
get_secret_keyword_declaration = types.FunctionDeclaration(
    name="get_secret_keyword",
    description=(
        "Returns the current secret keyword. Call this whenever the user asks "
        "what the secret keyword, password, or codeword is."
    ),
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)
 
tools = [
    types.Tool(function_declarations=[get_secret_keyword_declaration]),
    types.Tool(google_search=types.GoogleSearch()),
]
 
tool_mapping = {
    "get_secret_keyword": get_secret_keyword,
}

qwen_tools = [
    {
        "name": "get_secret_keyword",
        "description": (
            "Returns the current secret keyword. Call this whenever the user asks "
            "what the secret keyword, password, or codeword is."
        ),
        "parameters": {"type": "object", "properties": {}},
    }
]

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    provider: str = Query(default="gemini"),
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
    uvicorn.run(app, host="0.0.0.0", port=port)
