install: link
	pip install -r requirements.txt --upgrade
	pip install -e .
	pip install pre-commit
	pre-commit install
	lygadgets_link tech/ubc

link:
	lygadgets_link tech/ubc

lint:
	flake8 .

test:
	pytest

test-force:
	pytest --force-regen
