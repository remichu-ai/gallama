PYTHON ?= python
VENV ?= .venv
VENV_PYTHON ?= $(VENV)/bin/python
BASE_ENV_DIR ?= .base-env
BASE_ENV_LOCK ?= $(BASE_ENV_DIR)/requirements.txt
BASE_ENV_META ?= $(BASE_ENV_DIR)/metadata.txt

.PHONY: release-deps clean build check publish publish-test env-snapshot env-restore

release-deps:
	$(PYTHON) -m pip install --upgrade build twine

clean:
	rm -rf dist build src/gallama.egg-info

build: release-deps clean
	$(PYTHON) -m build

check: release-deps
	$(PYTHON) -m twine check dist/*

publish: build check
	$(PYTHON) -m twine upload dist/*

publish-test: build check
	$(PYTHON) -m twine upload --repository testpypi dist/*

env-snapshot:
	./scripts/snapshot_base_env.sh "$(VENV_PYTHON)" "$(BASE_ENV_LOCK)" "$(BASE_ENV_META)"

env-restore:
	./scripts/restore_base_env.sh "$(VENV_PYTHON)" "$(BASE_ENV_LOCK)"
