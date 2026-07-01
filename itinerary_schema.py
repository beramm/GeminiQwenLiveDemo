# itinerary_schema.py
#
# Single source of truth for the travel itinerary structured-output schema.
# Used three ways: jsonschema.validate() directly, Qwen's response_format.json_schema
# (as-is, it's already OpenAI-compatible JSON Schema), and converted to Gemini's
# types.Schema via dict_to_gemini_schema(). One definition, three consumers — avoids
# hand-duplicated schemas drifting out of sync between providers.

from google.genai import types

_COST = {
    "type": "object",
    "properties": {
        "amount": {"type": "number", "description": "Numeric cost estimate."},
        "currency": {"type": "string", "description": "ISO 4217 currency code, e.g. USD."},
    },
    "required": ["amount", "currency"],
    "additionalProperties": False,
}

_LOCATION = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "address": {"type": "string"},
        "category": {"type": "string", "description": "e.g. restaurant, museum, park, transport hub."},
    },
    "required": ["name", "address", "category"],
    "additionalProperties": False,
}

_ACTIVITY = {
    "type": "object",
    "properties": {
        "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "evening", "night"]},
        "start_time": {"type": "string", "description": "24h HH:MM, e.g. 09:30."},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "location": _LOCATION,
        "estimated_cost": _COST,
        "tips": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["time_of_day", "start_time", "title", "description", "location", "estimated_cost", "tips"],
    "additionalProperties": False,
}

_ACCOMMODATION = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string", "description": "e.g. hotel, hostel, apartment."},
        "estimated_cost_per_night": _COST,
    },
    "required": ["name", "type", "estimated_cost_per_night"],
    "additionalProperties": False,
}

_MEALS = {
    "type": "object",
    "properties": {
        "breakfast": {"type": "string"},
        "lunch": {"type": "string"},
        "dinner": {"type": "string"},
    },
    "required": ["breakfast", "lunch", "dinner"],
    "additionalProperties": False,
}

_DAY = {
    "type": "object",
    "properties": {
        "day_number": {"type": "integer"},
        "theme": {"type": "string"},
        "meals": _MEALS,
        "accommodation": _ACCOMMODATION,
        "activities": {"type": "array", "items": _ACTIVITY},
    },
    "required": ["day_number", "theme", "meals", "accommodation", "activities"],
    "additionalProperties": False,
}

ITINERARY_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "duration_days": {"type": "integer"},
        "preference": {"type": "string"},
        "summary": {"type": "string"},
        "total_estimated_budget": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "currency": {"type": "string"},
                "breakdown_notes": {"type": "string"},
            },
            "required": ["amount", "currency", "breakdown_notes"],
            "additionalProperties": False,
        },
        "packing_suggestions": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "days": {"type": "array", "items": _DAY},
    },
    "required": [
        "destination", "duration_days", "preference", "summary",
        "total_estimated_budget", "packing_suggestions", "notes", "days",
    ],
    "additionalProperties": False,
}


def dict_to_gemini_schema(schema: dict) -> types.Schema:
    """Recursively convert a plain JSON Schema dict into google-genai types.Schema."""
    schema_type = schema.get("type", "object")

    if schema_type == "object":
        properties = {
            key: dict_to_gemini_schema(value)
            for key, value in schema.get("properties", {}).items()
        }
        return types.Schema(
            type=types.Type.OBJECT,
            properties=properties,
            required=schema.get("required", []),
        )

    if schema_type == "array":
        return types.Schema(
            type=types.Type.ARRAY,
            items=dict_to_gemini_schema(schema["items"]),
        )

    if schema_type == "string":
        kwargs = {"type": types.Type.STRING}
        if "enum" in schema:
            kwargs["enum"] = schema["enum"]
        if "description" in schema:
            kwargs["description"] = schema["description"]
        return types.Schema(**kwargs)

    if schema_type == "integer":
        return types.Schema(type=types.Type.INTEGER)

    if schema_type == "number":
        return types.Schema(type=types.Type.NUMBER)

    raise ValueError(f"Unsupported JSON Schema type for Gemini conversion: {schema_type}")
