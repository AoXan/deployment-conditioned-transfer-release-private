from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


OLD_V2_SUFFIX = "universal_weather_full_campaign_v2"
NEW_SUFFIX = "universal_weather_full_campaign_v2_remediation_v1"


def load_config(config_path: Path | str, schema_path: Path | str) -> dict:
    config_path, schema_path = Path(config_path), Path(schema_path)
    config = yaml.safe_load(config_path.read_text())
    schema = json.loads(schema_path.read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(f"{list(error.path)}: {error.message}" for error in errors)
        raise ValueError(f"CONFIG_SCHEMA_INVALID:{details}")
    output = Path(config["output_root"])
    if output.name == OLD_V2_SUFFIX:
        raise ValueError("OLD_V2_NAMESPACE_FORBIDDEN")
    if output.name != NEW_SUFFIX:
        raise ValueError("REMEDIATION_NAMESPACE_REQUIRED")
    return config
