# Reproducible pipeline. `make all` = full analysis; `make quick` = fast end-to-end check.
# Each target re-runs only when its inputs changed.

PY      := uv run python
Q       :=
ifeq ($(QUICK),1)
Q := --quick
endif

RAW     := data/raw/Clincial_annotations.csv data/raw/complete.csv
PROC    := data/processed/transfers.csv data/processed/embryos.csv
T       := results/tables

.PHONY: all quick setup data data-images audit test lint clean clean-all images explain

all: results/summary.md

quick:
	HEADROOM_QUICK=1 $(MAKE) all QUICK=1

setup:
	uv sync --group dev

data $(RAW):
	$(PY) scripts/00_download.py

data-images:
	$(PY) scripts/00_download.py --images

$(T)/audit.csv: $(RAW) src/headroom/audit.py src/headroom/cleaning.py
	$(PY) scripts/01_audit.py $(Q)

$(PROC): $(RAW) $(T)/audit.csv src/headroom/cleaning.py config.yaml
	$(PY) scripts/02_clean.py $(Q)

data/processed/predictions_embryos.csv $(T)/model_C.json: $(PROC) src/headroom/model.py \
                                                          src/headroom/pipeline.py config.yaml
	$(PY) scripts/03_model.py $(Q)

$(T)/strategies.csv: data/processed/predictions_embryos.csv src/headroom/strategies.py
	$(PY) scripts/04_strategies.py $(Q)

$(T)/choice_value.csv: data/processed/predictions_embryos.csv src/headroom/choice_value.py
	$(PY) scripts/05_choice_value.py $(Q)

$(T)/simulation_grid.csv: $(T)/strategies.csv $(T)/choice_value.csv src/headroom/simulation.py
	$(PY) scripts/06_simulation.py $(Q)

$(T)/observers.csv: data/processed/predictions_embryos.csv src/headroom/observers.py
	$(PY) scripts/07_observers.py $(Q)

$(T)/sensitivity.csv: $(T)/model_C.json src/headroom/sensitivity.py src/headroom/strategies.py
	$(PY) scripts/08_sensitivity.py $(Q)

results/summary.md: $(T)/strategies.csv $(T)/choice_value.csv $(T)/simulation_grid.csv \
                    $(T)/observers.csv $(T)/sensitivity.csv src/headroom/figures.py
	$(PY) scripts/09_figures.py $(Q)

# Optional step G (needs `uv sync --extra images` and `make data-images`).
images: $(T)/model_C.json
	$(PY) scripts/10_images.py $(Q)

test:
	uv run pytest

lint:
	uv run ruff check src scripts tests

clean:
	rm -rf data/processed/* results/tables/* results/figures/* results/logs/* results/summary.md

clean-all: clean
	find data/raw -mindepth 1 ! -name .gitkeep -delete

# Optional step H: explainability (after `make images`).
explain: $(T)/model_C.json
	$(PY) scripts/11_explain.py $(Q)
