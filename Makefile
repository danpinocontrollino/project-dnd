# True Lethality Engine — pipeline orchestration.
#
#   make retrain    # full pipeline: data -> train -> tests -> behavior suite
#   make app        # launch the Streamlit site
#   make help       # list all targets
#
# Every stage is deterministic (seeds fixed) and idempotent; artifacts land
# in the repo root and figures/.

PY := python3

.PHONY: help data train tune test verify retrain benchmark app cli clean

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

data:  ## Parse FIREBALL logs -> clean_aggregated_combat_data.csv (~2 min)
	$(PY) parse_fireball.py

train:  ## Train serving model with saved hyperparameters (~2 min)
	$(PY) initial_learn.py --no-tune --train-cr-predictor

tune:  ## Full Optuna hyperparameter search + train (~5-10 min)
	$(PY) initial_learn.py --trials 40 --train-cr-predictor

test:  ## Unit tests (parsing regexes, feature math, book math, physics guard)
	$(PY) -m pytest tests/ -q

verify:  ## Behavioral acceptance suite (domain axioms the model must satisfy)
	$(PY) behavior_suite.py

retrain: data train test verify  ## Full pipeline: data -> train -> test -> verify

benchmark:  ## Course-aligned model comparison (logistic, RFF kernel, MMD, GP)
	$(PY) model_comparison.py

battlecast:  ## Battlecast deathmatch grid (~15 min) + guard/mercy analysis
	node battlecast_bridge/run_grid.mjs --trials 2000
	$(PY) battlecast_bridge/analyze.py

slides:  ## Build the exam deck (presentation/slides.pdf); HTML twin needs no build
	cd presentation && pdflatex -interaction=nonstopmode slides.tex >/dev/null && pdflatex -interaction=nonstopmode slides.tex | tail -2

report:  ## Build the written report (presentation/report.pdf)
	cd presentation && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex | tail -1

present-data:  ## Regenerate real-model data for the interactive deck
	$(PY) presentation/make_interactive_data.py

present:  ## Serve the interactive exam deck at http://localhost:8765
	@echo "→ http://localhost:8765/presentation/slides_interactive.html"
	$(PY) -m http.server 8765

app:  ## Launch the Streamlit site
	$(PY) -m streamlit run app.py

cli:  ## Launch the terminal twin
	$(PY) fair_fight_finder.py

clean:  ## Remove caches (never touches data or models)
	rm -rf __pycache__ .pytest_cache tests/__pycache__
