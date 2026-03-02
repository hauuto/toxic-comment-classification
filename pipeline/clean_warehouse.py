"""Compatibility wrapper for the warehouse cleaning tool.

Prefer running:
  python -m pipeline.tools.clean_warehouse

This module keeps the older entrypoint working by delegating to the canonical
implementation in ``pipeline.tools.clean_warehouse``.
"""

from __future__ import annotations

from pipeline.tools.clean_warehouse import main


if __name__ == "__main__":
    raise SystemExit(main())
