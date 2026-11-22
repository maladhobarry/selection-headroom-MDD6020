"""Run analysis step: 07_observers (see src/headroom/pipeline.py:step_observers)."""

import argparse

from headroom.pipeline import step_observers

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    a = ap.parse_args()
    step_observers(quick=a.quick or None)
