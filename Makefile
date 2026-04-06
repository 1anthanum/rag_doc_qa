.PHONY: help install lint format test coverage demo smoke clean docker

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt

lint: ## Run ruff + black checks
	ruff check src/ tests/ app.py eval/ scripts/
	black --check src/ tests/ app.py eval/ scripts/

format: ## Auto-format code with black + ruff fix
	ruff check --fix src/ tests/ app.py eval/ scripts/
	black src/ tests/ app.py eval/ scripts/

test: ## Run all unit tests
	pytest tests/ -v

coverage: ## Run tests with coverage report (fail under 70%)
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=70

demo: ## Run interactive CLI demo (requires ANTHROPIC_API_KEY)
	python demo.py --all

demo-local: ## Run demo with Ollama (no API key needed)
	LLM_PROVIDER=ollama python demo.py --all

bench: ## Run retrieval-only benchmark (free, no LLM calls)
	python -m eval.benchmark --retrieval-only -v

bench-full: ## Run full benchmark with generation (requires API key)
	python -m eval.benchmark -v --output eval/report.md --json eval/results.json

app: ## Launch Streamlit web UI
	streamlit run app.py

api: ## Launch FastAPI server
	uvicorn src.api.endpoints:create_app --factory --reload

smoke: ## Run full smoke test suite
	bash scripts/smoke_test.sh

index-conversations: ## Index Digital Self sample conversations
	python scripts/index_conversations.py --input data/conversations/ --config configs/digital_self.yaml

docker: ## Build Docker image
	docker build -t rag-doc-qa .

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache coverage.xml htmlcov/
