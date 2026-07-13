"""Explicit, fingerprinted per-job Python runtime dispatch."""
from __future__ import annotations
import hashlib, json, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

class RuntimeBlocked(RuntimeError): pass

@dataclass(frozen=True)
class ResolvedRuntime:
    profile: str
    interpreter: Path
    python_version: str
    torch_version: str | None
    jsonschema_version: str | None
    fingerprint: str

def runtime_fingerprint(interpreter: Path, versions: dict[str,str|None]) -> str:
    payload={'interpreter':str(interpreter.resolve()),'versions':versions}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def resolve_job_runtime(campaign: dict[str,Any], job: dict[str,Any], *, root: Path) -> ResolvedRuntime:
    profile=str(job.get('runtime_profile',''))
    if not profile: raise RuntimeBlocked(f"RUNTIME_PROFILE_MISSING:{job.get('id')}")
    spec=campaign.get('runtimes',{}).get(profile)
    if not spec: raise RuntimeBlocked(f"RUNTIME_PROFILE_UNKNOWN:{profile}")
    interpreter=Path(spec['interpreter']); interpreter=interpreter if interpreter.is_absolute() else (root/interpreter).resolve()
    if not interpreter.is_file(): raise RuntimeBlocked(f"RUNTIME_INTERPRETER_MISSING:{interpreter}")
    probe="import json,platform,importlib.metadata as m\ndef v(name):\n try: return m.version(name)\n except m.PackageNotFoundError: return None\nprint(json.dumps({'python':platform.python_version(),'torch':v('torch'),'jsonschema':v('jsonschema')}))"
    try: versions=json.loads(subprocess.check_output([str(interpreter),'-c',probe],text=True))
    except (subprocess.CalledProcessError,json.JSONDecodeError) as exc: raise RuntimeBlocked(f"RUNTIME_PROBE_FAILED:{profile}:{exc}") from exc
    expected=spec.get('expected',{})
    for key,prefix in expected.items():
        if not str(versions.get(key) or '').startswith(str(prefix)): raise RuntimeBlocked(f"RUNTIME_VERSION_MISMATCH:{profile}:{key}")
    fp=runtime_fingerprint(interpreter,versions)
    return ResolvedRuntime(profile,interpreter,versions['python'],versions.get('torch'),versions.get('jsonschema'),fp)
