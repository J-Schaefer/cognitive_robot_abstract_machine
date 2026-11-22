#!/usr/bin/env python
"""
Runs the DAiSy retrace demo as a regression test.
"""

from __future__ import annotations

import traceback


def main() -> None:
    """
    Run the DAiSy retrace demo and exit non-zero with a traceback on failure.
    """
    try:
        import demo

        demo.main()
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
