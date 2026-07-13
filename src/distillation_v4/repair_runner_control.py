from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import CampaignConfig
from .repair_outer_preflight import (
    run_repair_outer_preflight,
)


def run_repair_preflight_only(
    *,
    config: CampaignConfig,
    output: Path,
) -> dict[str, Any]:
    repair = config.raw.get(
        "repair_contract",
        {},
    )

    if bool(
        repair.get(
            "formal_execution_enabled",
            False,
        )
    ):
        raise RuntimeError(
            "REPAIR_PREFLIGHT_REQUIRES_EXECUTION_DISABLED"
        )

    if bool(
        repair.get(
            "outer_release_enabled",
            False,
        )
    ) is not True:
        raise RuntimeError(
            "FORMAL_RUN_NOT_APPROVED"
        )

    return run_repair_outer_preflight(
        config=config,
        output=output,
    )
