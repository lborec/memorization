#!/bin/bash

export REQ_FILE = requirements/development.txt
export PROJECT_DIR = memorization
export TEST_DIR = tests

.PHONY: help
help:                  ## Show help messages and exit.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'


######################### MANAGE PYTHON ENVIRONMENT ####################################

.PHONY: venv
venv:                  ## Create local python venv for development
	if [[ -d ./venv ]]; then rm -rf venv; fi
	python -m venv venv
	. venv/bin/activate pip install --upgrade pip setuptools wheel build


.PHONY: install
install:               ## Install project locally in development mode - without dev tools
	. venv/bin/activate && pip install -e .


.PHONY: install-dev
install-dev:           ## Install project locally in development mode - with dev tools
	. venv/bin/activate && pip install -e ".[dev]"

######################### RUN TESTS ####################################

.PHONY: test
test:                  ## Run project tests using pytest
	. venv/bin/activate && python -m pytest ${TEST_DIR} -p no:warnings -s

.PHONY: qa
qa:                    ## Run both linter and pytest together
	make test
	make lint
