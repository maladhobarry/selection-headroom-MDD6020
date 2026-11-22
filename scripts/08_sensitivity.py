"""Run analysis step: 08_sensitivity (see src/headroom/pipeline.py:step_sensitivity)."""

import argparse

from headroom.pipeline import step_sensitivity

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    a = ap.parse_args()
    step_sensitivity(quick=a.quick or None)
