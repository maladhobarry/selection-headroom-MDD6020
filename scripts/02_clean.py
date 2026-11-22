"""Run analysis step: 02_clean (see src/headroom/pipeline.py:step_clean)."""

import argparse

from headroom.pipeline import step_clean

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    a = ap.parse_args()
    step_clean(quick=a.quick or None)
