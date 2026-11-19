#!/usr/bin/env bash
# Use the frozen local CPU model environment from the Windows checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${RETRIEVAL_PYTHON:-$HOME/.venvs/sparsecl/bin/python}" -B -c '
from importlib.util import find_spec
from pathlib import Path
import sys

# The frozen WSL environment predates the API client. Import only its pure-Python
# dotenv dependency from the Windows project venv if it is missing in WSL.
site = Path.cwd() / ".venv/Lib/site-packages"
if find_spec("dotenv") is None and (site / "dotenv").is_dir():
    sys.path.append(str(site))
    try:
        import dotenv
    finally:
        sys.path.pop()
from memory_retrieval.cli import main
main()
' "$@"
