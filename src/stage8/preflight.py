"""Scientific preflight checks that must pass before a formal launch."""

from pathlib import Path


def formal_preflight(config: dict, *, root: Path) -> list[dict]:
    rows = []
    manifest = root / config.get("snapshot_manifest", "")
    rows.append({"check": "snapshot_manifest", "status": "PASS" if manifest.is_file() else "BLOCK", "details": str(manifest)})
    output = str(config.get("outputs", {}).get("root", "")).lower()
    rows.append({"check": "formal_namespace", "status": "BLOCK" if "smoke" in output else "PASS", "details": output})
    for route, protocol in config.get("route_protocols", {}).items():
        adapter = protocol.get("adapter_command")
        safe = isinstance(adapter, list) and adapter and not any("smoke" in str(part).lower() for part in adapter)
        rows.append({"check": f"route_adapter:{route}", "status": "PASS" if safe else "BLOCK", "details": adapter or "missing"})
    return rows
