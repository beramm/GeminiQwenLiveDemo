import asyncio
import base64
import inspect
import json
import logging
import time
import traceback

import websockets

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


class QwenOmni:
    """
    Handles the interaction with the Qwen-Omni Realtime API (DashScope).

    Mirrors the interface of GeminiLive so it can be swapped in transparently
    by main.py based on a `provider` query parameter.
    """

    def __init__(
        self,
        api_key,
        model,
        input_sample_rate=16000,
        base_url=DEFAULT_BASE_URL,
        voice="Ethan",
        instructions=None,
        tools=None,
        tool_mapping=None,
        mode="function_call",
    ):
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.base_url = base_url
        self.voice = voice
        self.mode = mode
        self.instructions = instructions or (
            "You are a helpful AI assistant. Keep your responses concise and friendly. "
            "You can see the user's camera or screen which is shared as realtime input images with you."
        )
        self.tools = tools or []
        self.tool_mapping = tool_mapping or {}

    async def _send_event(self, ws, event):
        event["event_id"] = "event_" + str(int(time.time() * 1000))
        await ws.send(json.dumps(event))

    async def start_session(
        self,
        audio_input_queue,
        video_input_queue,
        text_input_queue,
        audio_output_callback,
        audio_interrupt_callback=None,
    ):
        url = f"{self.base_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        logger.info(f"Connecting to Qwen Realtime with model={self.model}")
        try:
            async with websockets.connect(url, additional_headers=headers) as ws:
                logger.info("Qwen Realtime session opened successfully")

                session_config = {
                    "modalities": ["text", "audio"],
                    "voice": self.voice,
                    "instructions": self.instructions,
                    "input_audio_format": "pcm",
                    "output_audio_format": "pcm",
                    "input_audio_transcription": {"model": "gummy-realtime-v1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 500,
                        "silence_duration_ms": 800,
                    },
                    
                }

                if self.mode == "grounding":
                    session_config["enable_search"] = True
                    session_config["search_options"] = {"enable_source": True}
                elif self.tools:
                    session_config["tools"] = [
                        {
                            "type": "function",
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                        }
                        for t in self.tools
                    ]
                # After session_config send, add this:

                await self._send_event(ws, {"type": "session.update", "session": session_config})

                # Prime the audio buffer with 100ms of silence (16000 Hz, 16-bit PCM = 3200 bytes of zeros)
                silence = bytes(3200)
                silence_b64 = base64.b64encode(silence).decode("ascii")
                await self._send_event(ws, {
                    "type": "input_audio_buffer.append",
                    "audio": silence_b64,
                })
                logger.debug("Sent silent audio primer to satisfy Qwen audio-before-image ordering")

                event_queue = asyncio.Queue()

                # Track whether the model is currently producing a response so we
                # can issue response.cancel + emit "interrupted" when the user
                # starts speaking mid-response.
                state = {
                    "is_responding": False,
                    "current_response_id": None,
                    "audio_primed": False,  
                    "session_ready": False, 
                    "pending_tool_calls": {},

                }

                async def send_audio():
                    try:
                        while True:
                            chunk = await audio_input_queue.get()
                            # state["audio_primed"] = True
                            audio_b64 = base64.b64encode(chunk).decode("ascii")
                            await self._send_event(
                                ws,
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_b64,
                                },
                            )
                    except asyncio.CancelledError:
                        logger.debug("send_audio task cancelled")
                    except Exception as e:
                        logger.error(f"send_audio error: {e}\n{traceback.format_exc()}")

                async def send_video():
                    try:
                        while True:
                            chunk = await video_input_queue.get()
                            if not state["audio_primed"] or not state["session_ready"]:
                                logger.debug("Dropping image frame — audio not yet primed")
                                continue
                            logger.info(f"Sending image frame to Qwen: {len(chunk)} bytes")
                            image_b64 = base64.b64encode(chunk).decode("ascii")
                            await self._send_event(
                                ws,
                                {
                                    "type": "input_image_buffer.append",
                                    "image": image_b64,
                                },
                            )
                    except asyncio.CancelledError:
                        logger.debug("send_video task cancelled")
                    except Exception as e:
                        logger.error(f"send_video error: {e}\n{traceback.format_exc()}")

                async def send_text():
                    try:
                        while True:
                            text = await text_input_queue.get()
                            logger.info(f"Sending text to Qwen: {text}")
                            await self._send_event(
                                ws,
                                {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [
                                            {"type": "input_text", "text": text}
                                        ],
                                    },
                                },
                            )
                            await self._send_event(ws, {"type": "response.create"})
                    except asyncio.CancelledError:
                        logger.debug("send_text task cancelled")
                    except Exception as e:
                        logger.error(f"send_text error: {e}\n{traceback.format_exc()}")

                async def receive_loop():
                    try:
                        async for raw in ws:
                            try:
                                event = json.loads(raw)
                            except json.JSONDecodeError:
                                logger.warning(f"Non-JSON message from Qwen: {raw[:200]}")
                                continue

                            event_type = event.get("type", "")
                            logger.debug(f"Qwen event: {event_type}")

                            if event_type == "error":
                                err = event.get("error", {})
                                logger.error(f"Qwen error: {err}")
                                await event_queue.put(
                                    {"type": "error", "error": str(err)}
                                )
                                continue
                            
                            if event_type in ("session.created", "session.updated"):
                                sess = event.get("session", {})
                                logger.info(f"Qwen session ready: id={sess.get('id')}")
                                state["audio_primed"] = True   
                                state["session_ready"] = True
                                continue


                            if event_type == "response.created":
                                state["is_responding"] = True
                                state["current_response_id"] = event.get(
                                    "response", {}
                                ).get("id")
                                continue

                            # Tool call initiated — Qwen sends name + call_id here
                            if event_type == "response.output_item.added":
                                item = event.get("item", {})
                                if item.get("type") == "function_call":
                                    call_id = item.get("call_id", "")
                                    func_name = item.get("name", "")
                                    state["pending_tool_calls"][call_id] = {
                                        "name": func_name,
                                        "arguments_buf": "",
                                    }
                                continue

                            # Arguments stream in as deltas
                            if event_type == "response.function_call_arguments.delta":
                                call_id = event.get("call_id", "")
                                delta = event.get("delta", "")
                                if call_id in state["pending_tool_calls"]:
                                    state["pending_tool_calls"][call_id]["arguments_buf"] += delta
                                continue

                            # Arguments complete — execute and send result back
                            if event_type == "response.function_call_arguments.done":
                                call_id = event.get("call_id", "")
                                pending = state["pending_tool_calls"].pop(call_id, None)
                                if pending:
                                    func_name = pending["name"]
                                    try:
                                        args = json.loads(pending["arguments_buf"] or "{}")
                                    except json.JSONDecodeError:
                                        args = {}

                                    result = f"Error: tool '{func_name}' not found"
                                    if func_name in self.tool_mapping:
                                        try:
                                            tool_func = self.tool_mapping[func_name]
                                            if inspect.iscoroutinefunction(tool_func):
                                                result = await tool_func(**args)
                                            else:
                                                loop = asyncio.get_running_loop()
                                                result = await loop.run_in_executor(None, lambda: tool_func(**args))
                                        except Exception as e:
                                            result = f"Error: {e}"

                                    await event_queue.put({
                                        "type": "tool_call",
                                        "name": func_name,
                                        "args": args,
                                        "result": result,
                                    })

                                    # Send result back to Qwen so it can continue the response
                                    await self._send_event(ws, {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call_id,
                                            "output": json.dumps({"result": str(result)}),
                                        },
                                    })
                                    await self._send_event(ws, {"type": "response.create"})
                                continue

                            if event_type == "input_audio_buffer.speech_started":
                                if state["is_responding"]:
                                    if state["current_response_id"]:
                                        try:
                                            await self._send_event(
                                                ws, {"type": "response.cancel"}
                                            )
                                        except Exception as e:
                                            logger.warning(f"response.cancel failed: {e}")
                                    if audio_interrupt_callback:
                                        if inspect.iscoroutinefunction(
                                            audio_interrupt_callback
                                        ):
                                            await audio_interrupt_callback()
                                        else:
                                            audio_interrupt_callback()
                                    await event_queue.put({"type": "interrupted"})
                                    state["is_responding"] = False
                                    state["current_response_id"] = None
                                continue

                            if event_type == "response.audio.delta":
                                delta_b64 = event.get("delta", "")
                                if delta_b64:
                                    audio_bytes = base64.b64decode(delta_b64)
                                    if inspect.iscoroutinefunction(audio_output_callback):
                                        await audio_output_callback(audio_bytes)
                                    else:
                                        audio_output_callback(audio_bytes)
                                continue

                            if (
                                event_type
                                == "conversation.item.input_audio_transcription.completed"
                            ):
                                transcript = event.get("transcript", "")
                                if transcript:
                                    await event_queue.put(
                                        {"type": "user", "text": transcript}
                                    )
                                continue

                            if event_type == "response.audio_transcript.delta":
                                delta = event.get("delta", "")
                                if delta:
                                    await event_queue.put(
                                        {"type": "gemini", "text": delta}
                                    )
                                continue

                            if event_type == "response.done":
                                state["is_responding"] = False
                                state["current_response_id"] = None
                                await event_queue.put({"type": "turn_complete"})

                                usage = (
                                    event.get("response", {}).get("usage") or {}
                                )
                                if usage:
                                    usage_event = {
                                        "type": "usage_metadata",
                                        "input_tokens": usage.get("input_tokens", 0)
                                        or 0,
                                        "output_tokens": usage.get(
                                            "output_tokens", 0
                                        )
                                        or 0,
                                        "total_tokens": usage.get("total_tokens", 0)
                                        or 0,
                                    }
                                    in_details = usage.get("input_tokens_details")
                                    out_details = usage.get("output_tokens_details")
                                    if in_details:
                                        usage_event["input_details"] = [
                                            {"modality": k, "tokens": v}
                                            for k, v in in_details.items()
                                            if isinstance(v, int)
                                        ]
                                    if out_details:
                                        usage_event["output_details"] = [
                                            {"modality": k, "tokens": v}
                                            for k, v in out_details.items()
                                            if isinstance(v, int)
                                        ]
                                    await event_queue.put(usage_event)
                                continue

                    except asyncio.CancelledError:
                        logger.debug("receive_loop task cancelled")
                    except websockets.ConnectionClosed as e:
                        logger.info(f"Qwen WebSocket closed: {e}")
                    except Exception as e:
                        logger.error(
                            f"receive_loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                        )
                        await event_queue.put(
                            {"type": "error", "error": f"{type(e).__name__}: {e}"}
                        )
                    finally:
                        logger.info("Qwen receive_loop exiting")
                        await event_queue.put(None)

                send_audio_task = asyncio.create_task(send_audio())
                send_video_task = asyncio.create_task(send_video())
                send_text_task = asyncio.create_task(send_text())
                receive_task = asyncio.create_task(receive_loop())

                try:
                    while True:
                        event = await event_queue.get()
                        if event is None:
                            break
                        if isinstance(event, dict) and event.get("type") == "error":
                            yield event
                            break
                        yield event
                finally:
                    logger.info("Cleaning up Qwen Realtime session tasks")
                    send_audio_task.cancel()
                    send_video_task.cancel()
                    send_text_task.cancel()
                    receive_task.cancel()
        except Exception as e:
            logger.error(
                f"Qwen Realtime session error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            raise
        finally:
            logger.info("Qwen Realtime session closed")
