"""Run analysis step: 04_strategies (see src/headroom/pipeline.py:step_strategies)."""

import argparse

from headroom.pipeline import step_strategies

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    a = ap.parse_args()
    step_strategies(quick=a.quick or None)
