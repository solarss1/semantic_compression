# Семантичне стиснення зображень — desktop-застосунок

.PHONY: help setup setup-sam data app clean test

VENV   ?= .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

help:
	@echo ""
	@echo "  make setup     — venv + базові залежності + U²-Net"
	@echo "  make setup-sam — Ultralytics SAM (опційно, для кліків/bbox)"
	@echo "  make data      — завантажити PNG-зразки"
	@echo "  make app       — запустити desktop-застосунок"
	@echo "  make clean     — очистити .cache/ та __pycache__"
	@echo "  make test      — pytest"
	@echo ""

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Завантаження U²-Net (потрібен інтернет)…"
	$(PYTHON) -m src.roi.u2net_setup

setup-sam:
	@test -x $(PYTHON) || $(MAKE) setup
	$(PIP) install -r requirements-sam.txt
	$(PIP) install -e ".[sam]"
	$(PYTHON) -m src.roi.ultralytics_sam_setup --download

data:
	$(PYTHON) scripts/download_samples.py

app:
	$(PYTHON) -m src.app

clean:
	@echo "Очищення .cache/ …"
	@rm -rf .cache checkpoints/U-2-Net-master.zip
	@echo "Очищення __pycache__ …"
	@find . \( -path './.venv' -o -path './.git' -o -path './checkpoints/U-2-Net' -o -path './data/samples' \) -prune -o \
		-type d \( -name '__pycache__' -o -name '.pytest_cache' \) -exec rm -rf {} + 2>/dev/null; true
	@find . \( -path './.venv' -o -path './.git' \) -prune -o \
		-type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null; true
	@echo "Готово."

test:
	@$(PYTHON) -c "import pytest" 2>/dev/null || $(PIP) install pytest
	$(PYTHON) -m pytest
