install:
	bash install.sh

link:
	lygadgets_link lygadgets
	lygadgets_link tech/ubc

lint:
	flake8 .

test:
	pytest

test-force:
	pytest --force-regen

cov:
	pytest --cov=ubc

annotations:
	pip install pytest-monkeytype
	py.test --monkeytype-output=./monkeytype.sqlite3
	fish add_types.fish
