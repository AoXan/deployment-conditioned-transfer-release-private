#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.distillation_program.next_runner import cli_main
from src.distillation_program.next_execution import execute_program
if __name__ == "__main__":
    raise SystemExit(cli_main("privileged_ablation", execute_program))
