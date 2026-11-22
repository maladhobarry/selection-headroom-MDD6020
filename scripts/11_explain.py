"""Run analysis step: 11_explain (see src/headroom/pipeline.py:step_explain)."""

import argparse

from headroom.pipeline import step_explain

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random weights (offline smoke test only)")
    a = ap.parse_args()
    step_explain(quick=a.quick or None, pretrained=not a.no_pretrained)
