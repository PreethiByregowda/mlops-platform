# Pipeline (root) and inference (web_service/) each have their own locked
# environment; these targets drive both.

.PHONY: setup lint format test

setup:  ## install both environments exactly as locked
	pipenv sync --dev
	$(MAKE) -C web_service setup

lint:  ## ruff, same rules as CI
	pipenv run ruff check .

format:  ## sort imports and format code
	pipenv run isort .
	pipenv run black .

test:  ## pipeline suite, then inference suite
	pipenv run python -m pytest tests
	$(MAKE) -C web_service test
