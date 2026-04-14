PY ?= python
PORT ?= 8000
IMAGE ?= runhaoli-creator/dmapo:local

.PHONY: help install lint serve docker async-candidates pipeline clean

help:
	@echo "make install          - install package + requirements"
	@echo "make lint             - ruff check (syntax + style)"
	@echo "make serve            - run FastAPI judge API on :$(PORT)"
	@echo "make docker           - build local Docker image"
	@echo "make async-candidates - run async on-policy candidate generation"
	@echo "make pipeline         - run full 6-stage DMAPO pipeline"
	@echo "make clean            - remove caches and build artifacts"

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

lint:
	ruff check src serve scripts

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port $(PORT) --reload

docker:
	docker build -t $(IMAGE) .

async-candidates:
	$(PY) scripts/async_candidate_gen.py \
		--prompts data/processed/all_prompts.jsonl \
		--output data/processed/candidates.jsonl \
		--model Qwen/Qwen2.5-7B-Instruct \
		--k 4 --concurrency 32

pipeline:
	bash scripts/run_pipeline.sh

clean:
	rm -rf .ruff_cache .pytest_cache **/__pycache__ *.egg-info dist build
