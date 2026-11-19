"""Run the six frozen baselines with the shared experiment protocol."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_retrieval.cli import run_cli
from memory_retrieval.experiment import BASELINES


if __name__ == "__main__":
    run_cli(default_methods=BASELINES)
