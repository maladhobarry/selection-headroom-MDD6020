"""Run analysis step: 06_simulation (see src/headroom/pipeline.py:step_simulation)."""

import argparse

from headroom.pipeline import step_simulation

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    a = ap.parse_args()
    step_simulation(quick=a.quick or None)
