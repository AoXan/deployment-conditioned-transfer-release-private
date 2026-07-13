from __future__ import annotations

import re
from typing import Any


TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
}


def validate_json_schema(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate the JSON-Schema subset used by the campaign contract."""
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type:
        py_type = TYPE_MAP.get(expected_type)
        if py_type is None:
            return [f"{path}:unsupported_schema_type:{expected_type}"]
        if not isinstance(instance, py_type) or (expected_type == "integer" and isinstance(instance, bool)):
            return [f"{path}:expected_{expected_type}"]
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}:expected_const:{schema['const']}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}:not_in_enum")
    if isinstance(instance, str) and schema.get("pattern") and not re.search(schema["pattern"], instance):
        errors.append(f"{path}:pattern_mismatch")
    if isinstance(instance, list):
        if len(instance) < int(schema.get("minItems", 0)):
            errors.append(f"{path}:too_few_items")
        item_schema = schema.get("items")
        if item_schema:
            for index, value in enumerate(instance):
                errors.extend(validate_json_schema(value, item_schema, f"{path}[{index}]"))
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}:missing_required:{key}")
        for key, child_schema in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(validate_json_schema(instance[key], child_schema, f"{path}.{key}"))
    return errors
