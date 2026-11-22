"""Run analysis step: 01_audit (see src/headroom/pipeline.py:step_audit)."""

import argparse

from headroom.pipeline import step_audit

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    a = ap.parse_args()
    step_audit(quick=a.quick or None)
