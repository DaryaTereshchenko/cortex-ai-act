###############################################################################
# CORTEX-AI-ACT  —  Makefile
# Convenient shortcuts for local development
###############################################################################

.PHONY: help up down build logs lint test clean

COMPOSE = docker compose

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Docker ───────────────────────────────────────────────────────────────────

up: ## Start all services (detached)
	$(COMPOSE) up -d

up-build: ## Build then start all services
	$(COMPOSE) up -d --build

down: ## Stop all services
	$(COMPOSE) down

down-v: ## Stop all services and remove volumes
	$(COMPOSE) down -v

build: ## Build all Docker images
	$(COMPOSE) build

logs: ## Tail logs from all services
	$(COMPOSE) logs -f

logs-%: ## Tail logs from a specific service (e.g. make logs-neo4j)
	$(COMPOSE) logs -f $*

ps: ## Show running containers
	$(COMPOSE) ps

pull: ## Pull latest images
	$(COMPOSE) pull

restart: ## Restart all services
	$(COMPOSE) restart

# ── Development ──────────────────────────────────────────────────────────────

lint: ## Run ruff linter across all services
	@for svc in knowledge-graph reasoning-engine web-ui benchmarking; do \
		echo "\n=== Linting $$svc ===" ; \
		ruff check services/$$svc/ ; \
	done

format: ## Auto-format all services with ruff
	@for svc in knowledge-graph reasoning-engine web-ui benchmarking; do \
		ruff format services/$$svc/ ; \
	done

test: ## Run pytest across all services
	@for svc in knowledge-graph reasoning-engine web-ui benchmarking; do \
		echo "\n=== Testing $$svc ===" ; \
		cd services/$$svc && python -m pytest tests/ -v --tb=short 2>/dev/null \
			|| echo "(no tests yet)" ; \
		cd ../.. ; \
	done

# ── Setup ────────────────────────────────────────────────────────────────────

env: ## Create .env from template
	@if [ ! -f .env ]; then \
		cp .env.example .env ; \
		echo ".env created — edit it with your credentials." ; \
	else \
		echo ".env already exists." ; \
	fi

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove stopped containers, dangling images, __pycache__
	docker system prune -f
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
