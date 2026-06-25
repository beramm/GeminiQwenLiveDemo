# gemini_multimodal.py

import logging
import inspect

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiMultimodal:
    """One-shot Gemini multimodal client with agentic function-call loop."""

    MAX_TOOL_ROUNDS = 5

    def __init__(self, api_key: str, tools=None, tool_mapping=None):
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or []
        self.tool_mapping = tool_mapping or {}

    async def generate(
        self,
        model: str,
        prompt: str = "",
        media: dict | None = None,
        audio: dict | None = None,
        history: list | None = None,
    ) -> dict:
        # ── Build initial parts ──────────────────────────────────────────────
        parts: list[types.Part] = []

        if prompt := (prompt or "").strip():
            parts.append(types.Part.from_text(text=prompt))

        for blob in (media, audio):
            if blob:
                parts.append(
                    types.Part.from_bytes(
                        data=blob["data"],
                        mime_type=blob["mime_type"],
                    )
                )

        if not parts:
            raise ValueError("Provide text, a recording, or an uploaded media file.")

        # ── Config ───────────────────────────────────────────────────────────
        config = types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful multimodal AI assistant. "
                "Answer concisely and directly based on the user's text, "
                "recorded audio, uploaded image, or uploaded video. "
                "Use available tools when they are relevant."
            ),
            tools=self.tools,
        )

        logger.info("Starting multimodal generate | model=%s parts=%d", model, len(parts))

        # ── Create chat session (no persisted history) ───────────────────────
        # chat = self.client.aio.chats.create(model=model, config=config)

        chat = self.client.aio.chats.create(
            model=model,
            config=config,
            history=[
                types.Content(
                    role=turn["role"],
                    parts=[
                        types.Part.from_text(text=p["text"])
                        for p in turn.get("parts", [])
                        if p.get("text")
                    ]
                )
                for turn in (history or [])
            ]
        )

        # ── First turn ───────────────────────────────────────────────────────
        response = await chat.send_message(parts)
        logger.info("Initial response received | model=%s text=%s", model, response.text)

        tool_calls: list[dict] = []

        # ── Agentic tool-call loop ───────────────────────────────────────────
        for _round in range(self.MAX_TOOL_ROUNDS):
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                logger.info("No function calls found")
                break 

            function_responses: list[types.Part] = []

            for fc in function_calls:
                func_name = fc.name
                args = fc.args or {}
                logger.info("Tool call [%d]: %s  args=%s", _round + 1, func_name, args)

                # Execute the tool
                if func_name not in self.tool_mapping:
                    result = f"Error: unknown tool '{func_name}'"
                else:
                    try:
                        tool_func = self.tool_mapping[func_name]
                        if inspect.iscoroutinefunction(tool_func):
                            result = await tool_func(**args)
                        else:
                            result = tool_func(**args)
                    except Exception as exc:
                        result = f"Error: {exc}"
                        logger.exception("Tool '%s' raised an exception", func_name)

                tool_calls.append({"name": func_name, "args": args, "result": result})
 
                function_responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            id=getattr(fc, "id", None),
                            response={"result": result},
                        )
                    )
                )

            # Send all tool results back in one turn
            response = await chat.send_message(function_responses)

        # ── Extract usage ────────────────────────────────────────────────────
        usage_meta = getattr(response, "usage_metadata", None)
        usage = None
        if usage_meta:
            usage = {
                "input_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                "output_tokens": (
                    getattr(usage_meta, "candidates_token_count", 0)
                    or getattr(usage_meta, "response_token_count", 0)
                    or 0
                ),
                "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
            }

        return {
            "model": model,
            "text": response.text or "",
            "usage": usage,
            "tool_calls": tool_calls,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_function_calls(self, response) -> list:
        """Deduplicated function calls from a response object."""
        calls: list = []
        seen: set = set()

        def _add(fc):
            key = (getattr(fc, "id", None), fc.name, str(getattr(fc, "args", None)))
            if key not in seen:
                seen.add(key)
                calls.append(fc)

        # Top-level shortcut Gemini sometimes provides
        for fc in getattr(response, "function_calls", None) or []:
            _add(fc)

        # Walk candidates → content → parts (covers all cases)
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                if fc:
                    _add(fc)

        return calls
    
    
    