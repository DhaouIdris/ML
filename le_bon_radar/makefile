.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check ingest train app clean

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Installe les dépendances et les hooks pre-commit
	poetry install
	poetry run pre-commit install

lint: ## Vérifie le code avec Ruff
	poetry run ruff check src tests

format: ## Formate le code et corrige les erreurs automatiques
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

typecheck: ## Vérifie le typage statique avec mypy
	poetry run mypy src

test: ## Lance les tests avec couverture
	poetry run pytest

check: lint typecheck test ## Pipeline qualité complet

ingest: ## Récupère et nettoie les annonces (étape 2)
	poetry run python -m le_bon_radar.pipelines.ingest

train: ## Entraîne le modèle de prix (étape 3)
	poetry run python -m le_bon_radar.pipelines.train

app: ## Lance l'application Streamlit (étape 5)
	poetry run streamlit run src/le_bon_radar/app/main.py

clean: ## Supprime les caches et artefacts temporaires
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +