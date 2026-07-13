from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


COMMANDS = (
    "validate",
    "data-audit",
    "preprocess",
    "reproduce-primary",
    "evaluate-checkpoints",
    "reproduce-diagnostics",
    "reproduce-cases",
    "reproduce-apsim",
    "render-paper-assets",
    "render-paper-tables",
    "numeric-integrity",
    "verify-publication",
)


def repository_root() -> Path:
    working_directory = Path.cwd().resolve()
    if (working_directory / "configs/publication_repro.yaml").is_file():
        return working_directory
    return Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="agritech_repro",
        description="Reproduce and validate the deployment-conditioned transfer paper.",
    )
    sub = value.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        item = sub.add_parser(name, help=f"Run the {name} workflow.")
        item.add_argument("--config", type=Path, default=Path("configs/publication_repro.yaml"))
        item.add_argument("--root", type=Path, default=None, help="Repository root; inferred by default.")
        item.add_argument("--data-root", type=Path, default=None, help="Root containing acquired datasets.")
        item.add_argument("--checkpoint-root", type=Path, default=None, help="Root containing optional checkpoints.")
        item.add_argument("--output-root", type=Path, default=Path("outputs/publication_repro"))
        item.add_argument("--fixture", action="store_true", help="Use the redistributable synthetic fixture.")
        item.add_argument("--dry-run", action="store_true", help="Print the resolved action without executing it.")
        item.add_argument("--validate-only", action="store_true", help="Validate inputs without executing the workflow.")
        item.add_argument(
            "--confirm-full-campaign",
            help="Exact plan fingerprint required before a full primary campaign is started.",
        )
    return value


def emit(payload: dict[str, Any], code: int = 0) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return code


def load_context(args: argparse.Namespace) -> tuple[Path, dict[str, Any], Path, Path, Path]:
    root = (args.root or repository_root()).resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration missing: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    data_root = (args.data_root or Path(os.environ.get("AGRITECH_DATA_ROOT", root / "data/external"))).resolve()
    checkpoint_root = (
        args.checkpoint_root
        or Path(os.environ.get("AGRITECH_CHECKPOINT_ROOT", root / "artifacts/checkpoints"))
    ).resolve()
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    return root, config, data_root, checkpoint_root, output_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture_path(root: Path) -> Path:
    return root / "tests/fixtures/publication/primary_fixture.csv"


def required_repository_files(root: Path, config: dict[str, Any]) -> list[Path]:
    return [root / value for value in config["publication_assets"]["required_files"]]


def validate_repository(root: Path, config: dict[str, Any], fixture: bool) -> dict[str, Any]:
    missing = [str(path.relative_to(root)) for path in required_repository_files(root, config) if not path.is_file()]
    if fixture and not fixture_path(root).is_file():
        missing.append(str(fixture_path(root).relative_to(root)))
    return {
        "status": "PASS" if not missing else "BLOCKED",
        "fixture": fixture,
        "missing": missing,
        "python": sys.version.split()[0],
    }


def data_status(data_root: Path, config: dict[str, Any], fixture: bool, root: Path) -> list[dict[str, Any]]:
    if fixture:
        return [{"dataset": "synthetic_publication_fixture", "status": "AVAILABLE", "path": fixture_path(root)}]
    rows = []
    for name, spec in config["datasets"].items():
        path = data_root / spec["relative_path"]
        rows.append(
            {
                "dataset": name,
                "status": "AVAILABLE" if path.exists() else "MISSING",
                "path": path,
                "redistribution": spec["redistribution"],
                "required_for": spec["required_for"],
            }
        )
    return rows


def fixture_metrics(root: Path, output_root: Path, write: bool) -> dict[str, Any]:
    frame = pd.read_csv(fixture_path(root))
    required = {"sample_id", "year", "group", "y_true", "y_pred"}
    if not required.issubset(frame):
        raise ValueError(f"fixture missing columns: {sorted(required - set(frame))}")
    error = frame.y_true - frame.y_pred
    payload = {
        "n": int(len(frame)),
        "mae": float(error.abs().mean()),
        "rmse": float((error.pow(2).mean()) ** 0.5),
        "sample_unit": "synthetic_region_year",
    }
    if write:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "fixture_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_checked(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def publication_environment(data_root: Path, checkpoint_root: Path, output_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "AGRITECH_DATA_ROOT": str(data_root),
            "AGRITECH_CHECKPOINT_ROOT": str(checkpoint_root),
            "AGRITECH_FORMAL_ROOT": str(output_root / "formal_assets"),
            "PYTHONPATH": str(repository_root() / "src"),
        }
    )
    return env


def plan(command: str, config: dict[str, Any], root: Path, output_root: Path) -> dict[str, Any]:
    workflow = config["workflows"].get(command, {})
    return {
        "status": "DRY_RUN",
        "workflow": command,
        "executed": False,
        "entrypoint": workflow.get("entrypoint"),
        "output_root": output_root,
        "root": root,
    }


def require_data(rows: list[dict[str, Any]], workflow: str) -> list[dict[str, Any]]:
    needed = [row for row in rows if workflow in row.get("required_for", [])]
    missing = [row for row in needed if row["status"] != "AVAILABLE"]
    if missing:
        names = ", ".join(row["dataset"] for row in missing)
        raise FileNotFoundError(f"missing authorized data for {workflow}: {names}")
    return needed


def execute_workflow(args: argparse.Namespace) -> int:
    try:
        root, config, data_root, checkpoint_root, output_root = load_context(args)
        if args.dry_run:
            return emit(plan(args.command, config, root, output_root))
        if args.command == "validate":
            payload = validate_repository(root, config, args.fixture)
            return emit(payload, 0 if payload["status"] == "PASS" else 2)
        rows = data_status(data_root, config, args.fixture, root)
        if args.command == "data-audit":
            blocked = any(row["status"] == "MISSING" for row in rows if not args.fixture)
            return emit({"status": "BLOCKED" if blocked else "PASS", "datasets": rows}, 2 if blocked else 0)
        if args.fixture:
            metrics = fixture_metrics(root, output_root / args.command, not args.validate_only)
            return emit({"status": "PASS", "fixture": True, "workflow": args.command, "metrics": metrics})
        if args.command in {"preprocess", "reproduce-primary", "reproduce-cases"}:
            require_data(rows, args.command)
        if args.command == "preprocess":
            native = root / config["workflows"]["reproduce-primary"]["native_config"]
            resolved = output_root / "configs/stage8_v4_australian.resolved.yaml"
            prepare = [
                sys.executable,
                str(root / config["workflows"]["preprocess"]["entrypoint"]),
                "--cybench-root",
                str(data_root / "cybench/raw"),
                "--output-root",
                str(data_root / "cybench"),
            ]
            if args.validate_only:
                prepare.append("--validate-only")
            run_checked(prepare, root, publication_environment(data_root, checkpoint_root, output_root))
            command = [
                sys.executable,
                str(root / "scripts/materialize_publication_config.py"),
                "--input",
                str(native),
                "--output",
                str(resolved),
            ]
            if args.validate_only:
                command.append("--validate-only")
            run_checked(command, root, publication_environment(data_root, checkpoint_root, output_root))
            return emit({"status": "PASS", "mode": "validate-only" if args.validate_only else "materialised", "config": resolved})
        if args.command == "reproduce-primary":
            workflow = config["workflows"][args.command]
            native = root / workflow["native_config"]
            resolved = output_root / "configs/stage8_v4_australian.resolved.yaml"
            env = publication_environment(data_root, checkpoint_root, output_root)
            run_checked(
                [sys.executable, str(root / "scripts/materialize_publication_config.py"), "--input", str(native), "--output", str(resolved)],
                root,
                env,
            )
            mode = "--dry-run" if args.validate_only else "--run-formal"
            command = [
                sys.executable,
                str(root / workflow["entrypoint"]),
                "--config",
                str(resolved),
                "--output-root",
                str(output_root / "primary"),
                mode,
            ]
            if not args.validate_only:
                if not args.confirm_full_campaign:
                    raise ValueError(
                        "full primary execution requires --confirm-full-campaign with the preflight plan fingerprint"
                    )
                command.extend(["--confirm-full-campaign", args.confirm_full_campaign])
            run_checked(
                command,
                root,
                env,
            )
            return emit({"status": "PASS", "mode": mode.removeprefix("--"), "config": resolved})
        if args.command == "evaluate-checkpoints":
            lineage = root / config["workflows"][args.command]["lineage"]
            replay = root / config["workflows"][args.command]["replay_validation"]
            missing = [str(path) for path in (lineage, replay) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing checkpoint evaluation assets: {missing}")
            return emit({"status": "PASS", "lineage_rows": len(pd.read_csv(lineage)), "replay_rows": len(pd.read_csv(replay))})
        if args.command == "reproduce-diagnostics":
            workflow = config["workflows"][args.command]
            script = root / workflow["entrypoint"]
            if args.validate_only:
                required = [root / path for path in workflow["validation_assets"]]
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"missing frozen diagnostic output: {missing}")
                counts = {str(path.relative_to(root)): len(pd.read_csv(path)) for path in required if path.suffix == ".csv"}
                return emit({"status": "PASS", "mode": "frozen-output-validation", "rows": counts})
            else:
                replay_root = os.environ.get("AGRITECH_FORMAL_REPLAY_ROOT")
                primary_view = os.environ.get("AGRITECH_PRIMARY_VIEW")
                if not replay_root or not primary_view:
                    raise FileNotFoundError(
                        "full diagnostics require AGRITECH_FORMAL_REPLAY_ROOT and AGRITECH_PRIMARY_VIEW"
                    )
                env = publication_environment(data_root, checkpoint_root, output_root)
                env.update(
                    {
                        "FORMAL_CODE_ROOT": str(root),
                        "AGRITECH_FORMAL_REPLAY_ROOT": replay_root,
                        "AGRITECH_PRIMARY_VIEW": primary_view,
                    }
                )
                run_checked(
                    [
                        sys.executable,
                        str(script),
                        "--worktree",
                        str(root),
                        "--formal-code",
                        str(root),
                        "--output",
                        str(output_root / "diagnostics"),
                        "--shap-permutations",
                        "64",
                        "--background-rows",
                        "32",
                        "--ig-steps",
                        "64",
                        "--perm-reps",
                        "4999",
                        "--random-seed",
                        "20260709",
                    ],
                    root,
                    env,
                )
            return emit({"status": "PASS", "entrypoint": script})
        if args.command == "reproduce-cases":
            return emit({"status": "PASS", "mode": "validate-only", "datasets": rows})
        if args.command == "reproduce-apsim":
            required = [root / path for path in config["workflows"][args.command]["required_outputs"]]
            if args.validate_only:
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"missing APSIM outputs: {missing}")
            else:
                for script in config["workflows"][args.command]["commands"]:
                    run_checked([sys.executable, str(root / script)] if script.endswith(".py") else [str(root / script)], root)
            return emit({"status": "PASS", "workflow": args.command})
        if args.command == "render-paper-assets":
            script = root / config["workflows"][args.command]["entrypoint"]
            flag = "--validate-only" if args.validate_only else "--write"
            run_checked([sys.executable, str(script), "--root", str(root), flag], root)
            return emit({"status": "PASS", "entrypoint": script})
        if args.command == "render-paper-tables":
            script = root / config["workflows"][args.command]["entrypoint"]
            command = [sys.executable, str(script), "--root", str(root), "--output", str(output_root / "tables")]
            if args.validate_only:
                command.append("--validate-only")
            run_checked(command, root)
            if not args.validate_only:
                run_checked([sys.executable, str(root / "experiments/apsim_comparison/render_supplement_tables.py")], root)
            return emit({"status": "PASS", "entrypoint": script, "output": output_root / "tables"})
        if args.command == "numeric-integrity":
            script = root / config["workflows"][args.command]["entrypoint"]
            command = [sys.executable, str(script), "--root", str(root), "--output", str(output_root / "numeric_integrity.json")]
            if args.validate_only:
                command.append("--validate-only")
            completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(completed.stderr or completed.stdout)
            return emit(json.loads(completed.stdout))
        if args.command == "verify-publication":
            validation = validate_repository(root, config, False)
            figure_script = root / config["workflows"]["render-paper-assets"]["entrypoint"]
            run_checked([sys.executable, str(figure_script), "--root", str(root), "--validate-only"], root)
            run_checked([sys.executable, str(root / "scripts/render_publication_tables.py"), "--root", str(root), "--validate-only"], root)
            run_checked([sys.executable, str(root / "scripts/verify_publication_numbers.py"), "--root", str(root), "--validate-only"], root)
            required = [root / path for path in config["publication_assets"]["verification_files"]]
            missing = [str(path) for path in required if not path.is_file()]
            status = "PASS" if validation["status"] == "PASS" and not missing else "BLOCKED"
            return emit({"status": status, "repository": validation, "missing": missing}, 0 if status == "PASS" else 2)
        raise ValueError(f"unsupported command: {args.command}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        return emit({"status": "BLOCKED", "workflow": args.command, "reason": str(error)}, 2)


def main() -> int:
    return execute_workflow(parser().parse_args())
