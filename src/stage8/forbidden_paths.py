"""Pure path policy; rejection never probes the candidate path."""
from __future__ import annotations
from pathlib import Path
from typing import Any

class ForbiddenPathError(ValueError): pass

def forbidden_reason(value: str | Path) -> str | None:
    text=str(value).replace('\\','/').lower()
    parts={p for p in text.split('/') if p}
    if 'nvt' in parts or any(p.startswith('nvt_') for p in parts): return 'NVT_ACCESS_FORBIDDEN'
    if ('g2f' in text or 'genomes_to_fields' in text) and '2024' in text and any(t in text for t in ('observed','target','yield')):
        return 'G2F_2024_OBSERVED_TARGET_ACCESS_FORBIDDEN'
    return None

def assert_path_allowed(value: str | Path) -> None:
    reason=forbidden_reason(value)
    if reason: raise ForbiddenPathError(reason)

def validate_campaign_paths(campaign: dict[str,Any]) -> list[str]:
    errors=[]
    for job in campaign.get('jobs',[]):
        for value in job.get('command',[]):
            reason=forbidden_reason(str(value))
            if reason: errors.append(f"{job.get('id')}:{reason}")
        for key in ('input_path','view_path','data_path','fold_path','feature_path'):
            if key in job:
                reason=forbidden_reason(job[key])
                if reason: errors.append(f"{job.get('id')}:{reason}")
    return errors
