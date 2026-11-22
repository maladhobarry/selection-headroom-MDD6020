"""Run analysis step: 03_model (see src/headroom/pipeline.py:step_model)."""

import argparse

from headroom.pipeline import step_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    a = ap.parse_args()
    step_model(quick=a.quick or None)
