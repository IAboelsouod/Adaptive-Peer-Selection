# Adaptive Peer Selection — build targets
.PHONY: install test baseline adaptive sweep figures clean smoke

# Flower 1.28 requires Python >= 3.10; prefer newest available interpreter.
PYTHON ?= $(shell command -v python3.13 python3.12 python3.11 python3 2>/dev/null | head -1)
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

NUM_ROUNDS_SMOKE ?= 3
NUM_CLIENTS_SMOKE ?= 8
NUM_ROUNDS_FIGURES ?= 8
NUM_CLIENTS_FIGURES ?= 12
NUM_ROUNDS_SWEEP ?= 10
NUM_ROUNDS_BASELINE ?= 10

install:
	@echo "Using interpreter: $(PYTHON)"
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required (Flower 1.28)'"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

smoke:
	@echo "==> Smoke: flat FedAvg ($(NUM_ROUNDS_SMOKE) rounds, $(NUM_CLIENTS_SMOKE) clients)"
	$(PY) -m src.run --strategy flat \
		--num-rounds $(NUM_ROUNDS_SMOKE) --num-clients $(NUM_CLIENTS_SMOKE) \
		--run-name smoke_flat
	@echo "==> Smoke: adaptive peer selection"
	$(PY) -m src.run --strategy adaptive --alpha 0.1 --global-sync-every 2 \
		--num-rounds $(NUM_ROUNDS_SMOKE) --num-clients $(NUM_CLIENTS_SMOKE) \
		--run-name smoke_adaptive

test:
	@echo "==> Unit tests"
	$(PY) -m pytest tests/ -q
	@$(MAKE) smoke

baseline:
	$(PY) -m src.run --strategy flat \
		--run-name baseline_flat --num-rounds $(NUM_ROUNDS_BASELINE)

adaptive:
	$(PY) -m src.run --strategy adaptive --alpha 0.1 --global-sync-every 2 \
		--run-name baseline_adaptive --num-rounds $(NUM_ROUNDS_BASELINE)

sweep:
	$(PY) scripts/experiments.py --reduced --num-rounds $(NUM_ROUNDS_SWEEP)
	$(PY) scripts/aggregate_results.py

figures:
	$(PY) scripts/generate_figures.py \
		--num-rounds $(NUM_ROUNDS_FIGURES) --num-clients $(NUM_CLIENTS_FIGURES)

clean:
	rm -rf $(VENV) .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
