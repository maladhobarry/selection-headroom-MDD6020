"""Run analysis step: 00_download (see src/headroom/pipeline.py:step_download)."""

import argparse

from headroom.pipeline import step_download

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small replicate counts (pipeline check)")
    ap.add_argument("--images", action="store_true", help="also download the image archive")
    a = ap.parse_args()
    step_download(images=a.images, quick=a.quick or None)
