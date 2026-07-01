# qwen_multimodal.py

import base64
import inspect
import json
import logging

import httpx

logger = logging.getLogger(__name__)

DASHSCOPE_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"




QWEN_MULTIMODAL_MODELS = {
    "qwen3.6-flash": "qwen3.6-flash",  # direct model ID, no alias needed
    "qwen3.5-omni-flash": "qwen3.5-omni-flash",  # omni: processes audio track inside video_url natively
}

# Omni models process audio embedded in video_url natively and require stream=True
# on every request (not specific to any one feature — DashScope mandates it for Omni).
QWEN_STREAMING_REQUIRED_MODELS = {"qwen3.5-omni-flash"}

SYSTEM_PROMPT = (
   "You are a helpful multimodal AI assistant. "
    "Answer concisely and directly based on the user's text, "
    "recorded audio, uploaded image, or uploaded video. "
    "Use available tools when they are relevant."
)


class QwenMultimodal:
    """One-shot Qwen multimodal client (DashScope OpenAI-compat) with agentic tool loop."""

    MAX_TOOL_ROUNDS = 5

    def __init__(self, api_key: str, tools: list | None = None, tool_mapping: dict | None = None):
        self.api_key = api_key
        self.tools = tools or []           
        self.tool_mapping = tool_mapping or {}

    async def generate(
        self,
        model: str,
        prompt: str = "",
        media: dict | None = None,
        audio: dict | None = None,
        history: list | None = None,
        mode: str = "function_call",        # "function_call" | "grounding"
    ) -> dict:
        api_model = QWEN_MULTIMODAL_MODELS.get(model, model)

        # ── Build user content parts ─────────────────────────────────────────
        user_content: list[dict] = []
        logger.info("Qwen generate | model=%s  mode=%s  prompt=%s  media=%s  audio=%s  history_len=%d",
                    api_model, mode, prompt[:50], bool(media), bool(audio), len(history or []))

        if media:
            mime = media["mime_type"]
            b64  = base64.b64encode(media["data"]).decode()
            if mime.startswith("image/"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            elif mime.startswith("video/"):
                user_content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{b64}"
                    },
                })
                
                
            else:
                logger.warning("Unsupported media type for Qwen: %s", mime)

        if audio:
            mime = audio["mime_type"]
            b64  = base64.b64encode(audio["data"]).decode()
            user_content.append({
                "type": "input_audio",
                "input_audio": {"data": b64, "format": mime.split("/")[-1]},
            })

        if prompt := (prompt or "").strip():
            user_content.append({"type": "text", "text": prompt})

        if not user_content:
            raise ValueError("Provide text, a recording, or an uploaded media file.")

        # ── Build message history ─────────────────────────────────────────────
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in (history or []):
            role = turn.get("role")
            if role == "model":
                role = "assistant"
            text = " ".join(
                p["text"] for p in turn.get("parts", []) if p.get("text")
            )
            if text:
                messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": user_content})

        # ── Build request kwargs ─────────────────────────────────────────────
        request: dict = {"model": api_model, "messages": messages}
        
        logger.info("request: %s", json.dumps(request, indent=2)[:5000])
        logger.info("content : %s", json.dumps(user_content, ensure_ascii=False)[:300])

        if mode == "grounding":
            request["enable_search"] = True
            logger.info("Qwen grounding enabled | model=%s", api_model)
        elif self.tools:
            request["tools"] = self.tools
            request["tool_choice"] = "auto"
            logger.info("Qwen function_call mode | model=%s tools=%d", api_model, len(self.tools))

        logger.info("Starting Qwen multimodal generate | model=%s", api_model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        tool_calls_log: list[dict] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            # ── First call ───────────────────────────────────────────────────
            response_data = await self._post(client, headers, request)
            logger.info("Qwen initial response | %s", json.dumps(response_data, ensure_ascii=False)[:300])

            # ── Agentic tool-call loop (function_call mode only) ─────────────
            if mode == "function_call":
                for _round in range(self.MAX_TOOL_ROUNDS):
                    choice  = response_data["choices"][0]
                    message = choice["message"]
                    calls   = message.get("tool_calls") or []

                    if not calls:
                        logger.info("No tool calls found")
                        break

                    # Append assistant turn with tool_calls
                    messages.append(message)

                    for tc in calls:
                        func_name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"].get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}

                        logger.info("Tool call [%d]: %s  args=%s", _round + 1, func_name, args)

                        if func_name not in self.tool_mapping:
                            result = f"Error: unknown tool '{func_name}'"
                        else:
                            try:
                                fn = self.tool_mapping[func_name]
                                result = await fn(**args) if inspect.iscoroutinefunction(fn) else fn(**args)
                            except Exception as exc:
                                result = f"Error: {exc}"
                                logger.exception("Tool '%s' raised", func_name)

                        tool_calls_log.append({"name": func_name, "args": args, "result": result})

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": str(result),
                        })

                    request["messages"] = messages
                    response_data = await self._post(client, headers, request)

        # ── Extract final text + usage ────────────────────────────────────────
        final_choice = response_data["choices"][0]
        text = (
            final_choice.get("message", {}).get("content")
            or final_choice.get("text")
            or ""
        )
        if isinstance(text, list):
            text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))

        raw_usage = response_data.get("usage", {})
        usage = {
            "input_tokens":  raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens":  raw_usage.get("total_tokens", 0),
        }

        return {
            "model": api_model,
            "text": text,
            "usage": usage,
            "tool_calls": tool_calls_log,
        }

    async def generate_itinerary(
        self,
        model: str,
        destination: str,
        days: int,
        preference: str,
    ) -> dict:
        """Single-call structured output: returns {text, usage} — text is raw model
        output, NOT pre-parsed. Validation happens centrally in main.py so both
        providers go through identical checks for a fair comparison."""
        from itinerary_schema import ITINERARY_SCHEMA

        api_model = QWEN_MULTIMODAL_MODELS.get(model, model)
        is_streaming_required = api_model in QWEN_STREAMING_REQUIRED_MODELS

        prompt = (
            f"Create a detailed {days}-day travel itinerary for {destination}. "
            f"Traveler preference: {preference}. "
            "Include realistic activities, meals, accommodation, and cost estimates for each day."
        )

        messages: list[dict] = [
            {
                "role": "system",
                # DashScope requires the literal word "json" somewhere in `messages`
                # to accept response_format (verified live: 400s otherwise with
                # "'messages' must contain the word 'json' in some form").
                "content": "You are a travel planning assistant that produces a detailed, realistic travel itinerary as a JSON object.",
            },
            {"role": "user", "content": prompt},
        ]

        json_schema_format = {
            "type": "json_schema",
            "json_schema": {"name": "TravelItinerary", "schema": ITINERARY_SCHEMA},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if is_streaming_required:
            request: dict = {
                "model": api_model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "modalities": ["text"],
                "response_format": json_schema_format,
            }
            logger.info("Starting Qwen Omni itinerary generate | model=%s destination=%s days=%s", api_model, destination, days)
            async with httpx.AsyncClient(timeout=120.0) as client:
                content, _tool_calls, raw_usage = await self._post_stream(client, headers, request)
            raw_usage = raw_usage or {}
        else:
            request = {
                "model": api_model,
                "messages": messages,
                "response_format": json_schema_format,
            }
            logger.info("Starting Qwen itinerary generate | model=%s destination=%s days=%s", api_model, destination, days)
            async with httpx.AsyncClient(timeout=60.0) as client:
                response_data = await self._post(client, headers, request)
            content = response_data["choices"][0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            raw_usage = response_data.get("usage", {})

        usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
        }

        return {
            "model": api_model,
            "text": content,
            "usage": usage,
        }

    async def _post(self, client: httpx.AsyncClient, headers: dict, payload: dict) -> dict:
        resp = await client.post(DASHSCOPE_API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            logger.error("DashScope error %d: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        return resp.json()

    async def _post_stream(
        self, client: httpx.AsyncClient, headers: dict, payload: dict
    ) -> tuple[str, list[dict], dict | None]:
        """Omni models require stream=True. Accumulates delta.content text, merged
        tool_calls (OpenAI-style streaming merge by index), and final usage."""
        full_content = ""
        usage = None
        tool_calls_acc: dict[int, dict] = {}
        async with client.stream("POST", DASHSCOPE_API_URL, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error("DashScope stream error %d: %s", resp.status_code, body)
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        full_content += piece
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls_acc.setdefault(
                            idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
        tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        return full_content, tool_calls, usage