install:
	pip install -r requirements.txt --upgrade
	pip install -e .
	pip install pre-commit
	pre-commit install

link:
	lygadgets_link lygadgets
	lygadgets_link tech/ubc

lint:
	flake8 .

test:
	pytest

test-force:
	pytest --force-regen
