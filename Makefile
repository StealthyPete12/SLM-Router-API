API_URL ?= http://localhost:8000

# ROUTER_API_KEY for `make dry-run` / `make demo`, and STACK_MODE, when .env exists.
-include .env
export ROUTER_API_KEY

# Which models answer (set STACK_MODE in .env, or `make up STACK_MODE=mock`):
#   real     Ollama + your cloud provider (the default)
#   mock     Ollama + a labelled mock cloud model: no API key needed
#   offline  mock local and cloud models: no API key and no downloaded models
# Mock answers, latencies and prices are simulated; never quote them as results.
STACK_MODE ?= real
COMPOSE_FILES_real := -f docker-compose.yml
COMPOSE_FILES_mock := $(COMPOSE_FILES_real) -f docker-compose.mock.yml
COMPOSE_FILES_offline := $(COMPOSE_FILES_mock) -f docker-compose.offline.yml
ifeq ($(origin COMPOSE_FILES_$(STACK_MODE)),undefined)
$(error STACK_MODE must be real, mock or offline, got '$(STACK_MODE)')
endif
COMPOSE ?= docker compose $(COMPOSE_FILES_$(STACK_MODE))
# The service the demo stops to show a real local-to-cloud fallback.
LOCAL_BACKEND := $(if $(filter offline,$(STACK_MODE)),mock-ollama,ollama)

.DEFAULT_GOAL := help
.PHONY: help install lint format test check up up-gpu down ps logs logs-api logs-ollama \
	logs-postgres logs-prometheus logs-grafana logs-playground models health dry-run \
	demo demo-fallback playground dashboard reset-data

help: ## List the available commands
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install Python dependencies into .venv with uv
	uv sync

lint: ## Check linting and formatting with ruff
	uv run ruff check .
	uv run ruff format --check .

format: ## Fix lint issues and format the code with ruff
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite (no Docker or models needed)
	uv run pytest

check: lint test ## Lint and test

up: ## Build and start the whole stack (STACK_MODE: real, mock or offline), wait until healthy
	$(COMPOSE) up -d --build --wait
	@echo "API http://localhost:8000 · playground http://localhost:8501 · Grafana http://localhost:3000 · Prometheus http://localhost:9090"

up-gpu: ## Same as up, with the NVIDIA GPU given to ollama
	$(COMPOSE) -f docker-compose.gpu.yml up -d --build --wait

down: ## Stop the stack (models, database and dashboards are kept in volumes)
	$(COMPOSE) down

ps: ## Show the state of every service
	$(COMPOSE) ps

logs: ## Follow the logs of every service
	$(COMPOSE) logs -f

logs-api: ## Follow the API logs (one JSON object per event)
	$(COMPOSE) logs -f api

logs-ollama: ## Follow the Ollama logs
	$(COMPOSE) logs -f ollama

logs-postgres: ## Follow the Postgres logs
	$(COMPOSE) logs -f postgres

logs-prometheus: ## Follow the Prometheus logs
	$(COMPOSE) logs -f prometheus

logs-grafana: ## Follow the Grafana logs
	$(COMPOSE) logs -f grafana

logs-playground: ## Follow the Streamlit playground logs
	$(COMPOSE) logs -f playground

models: ## Pull every local model listed in config/models.yaml (run after make up)
	@tags=$$($(COMPOSE) exec -T api python -m app.config) || exit 1; \
	for tag in $$tags; do \
		echo "==> ollama pull $$tag"; \
		$(COMPOSE) exec ollama ollama pull $$tag || exit 1; \
	done

health: ## Show API, Ollama and Postgres health
	@curl -fsS $(API_URL)/health; echo

dry-run: ## Route the acceptance prompts through the running API's free /v1/route
	uv run python -m scripts.acceptance --url $(API_URL)

demo: ## Load the demo prompts, then a real fallback (briefly stops the local backend)
	uv run python -m eval.load_demo --url $(API_URL) --set main
	@$(MAKE) --no-print-directory demo-fallback

demo-fallback: ## Stop the local backend, send the fallback prompts, start it again
	@echo "==> stopping $(LOCAL_BACKEND) to show the local-to-cloud fallback"
	$(COMPOSE) stop $(LOCAL_BACKEND)
	@uv run python -m eval.load_demo --url $(API_URL) --set fallback; status=$$?; \
	echo "==> starting $(LOCAL_BACKEND) again"; \
	$(COMPOSE) start $(LOCAL_BACKEND); \
	exit $$status

playground: ## Run the Streamlit playground on the host (the stack also serves it on :8501)
	uv run streamlit run ui/playground.py

dashboard: ## Rebuild the Grafana dashboard JSON from scripts/build_dashboard.py
	uv run python -m scripts.build_dashboard

reset-data: ## DESTRUCTIVE: delete stored requests, feedback, metrics and Grafana state (keeps models)
	@printf "This deletes the postgres-data, prometheus-data and grafana-data volumes. Type 'reset' to go on: "; \
	read answer; [ "$$answer" = "reset" ] || { echo "cancelled"; exit 1; }
	$(COMPOSE) down
	docker volume rm -f slm-router_postgres-data slm-router_prometheus-data slm-router_grafana-data
	@echo "Done. 'make up' recreates the schema from infra/postgres/init.sql."
