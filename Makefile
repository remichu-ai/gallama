PYTHON ?= python

.PHONY: release-deps clean build check publish publish-test

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
