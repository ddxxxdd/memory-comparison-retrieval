"""Run OCSP-QR or the frozen baselines on a dataset or selected scenarios."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_retrieval.cli import run_cli as main
from memory_retrieval.dataset import select_queries


if __name__ == "__main__":
    main()
