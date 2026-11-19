"""Run the OCSP-QR comparison and the violation-probes-only ablation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_retrieval.cli import run_cli
# The recorded ValidMem analysis imports the original probe contract and JSON helpers here.
from memory_retrieval.data import read_jsonl, write_json
from memory_retrieval.ocsp import ONLINE_ZERO_SHOT_BIPOLAR_PROMPT, TOP_KS, extract_online_bipolar_probes


if __name__ == "__main__":
    run_cli(default_methods=("bm25", "dense", "hybrid", "probes_only_no_ortho", "ocsp_qr"))
