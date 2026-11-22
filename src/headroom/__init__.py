"""Headroom of embryo selection: analysis package.

Modules follow the steps of the pre-registered analysis plan:

- ``download``      step 0: fetch raw files (Figshare + GitHub) and verify checksums
- ``cleaning``      step A: repair, link and filter the data
- ``audit``         step A: data-quality report
- ``model``         step B: embryo-level prognostic model of live birth
- ``strategies``    step C: g-computation of selection strategies (E1-E4)
- ``choice_value``  step F: live birth by number of blastocysts available (E7)
- ``simulation``    step D: ceiling reachable by a better predictor (E5)
- ``observers``     step E: inter-embryologist reproducibility of the choice (E6)
- ``sensitivity``   S1-S9
- ``figures``       tables and figures for the article
- ``images``        optional step G (image features)
"""

__version__ = "0.1.0"
