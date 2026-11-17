#!/usr/bin/env python3
"""Execute a project notebook in place and preserve it only after success."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Notebook to execute")
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook_path = args.notebook.resolve()
    project_root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
        allow_errors=False,
    )
    client.execute()
    temporary = notebook_path.with_suffix(notebook_path.suffix + ".tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(notebook_path)
    print(f"Executed and saved {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
