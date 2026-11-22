"""Run analysis step: 05_choice_value (see src/headroom/pipeline.py:step_choice_value)."""

import argparse

from headroom.pipeline import step_choice_value

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    a = ap.parse_args()
    step_choice_value(quick=a.quick or None)
