# Reproduit tout le projet : figures, benchmark, données réelles, tests et notebook.
PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src

.PHONY: all paysage benchmark figures reelles annexe complements tests notebook clean

all: tests paysage benchmark figures reelles annexe complements notebook

paysage:         ## Critère profilé (flip-flop) et conditionnement
	$(PYTHON) scripts/paysage.py

benchmark:       ## 12 méthodes × 120 jeux de données synthétiques (environ 5 min sur 12 cœurs)
	$(PYTHON) scripts/benchmark.py

figures:         ## Figures et tableau du benchmark
	$(PYTHON) scripts/figures_benchmark.py

reelles:         ## Estimation et incertitude sur les 12 patients réels
	$(PYTHON) scripts/donnees_reelles.py

annexe:          ## Second cas d'étude : données synthétiques de l'annexe du sujet
	$(PYTHON) scripts/donnees_annexe.py

complements:     ## Figures 08 à 13 (environ 6 min)
	$(PYTHON) scripts/figures_complementaires.py

tests:           ## Tests automatiques
	$(PYTHON) -m pytest -q

notebook:        ## Exécute le notebook de présentation
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/analyse.ipynb

clean:
	rm -rf resultats figures
