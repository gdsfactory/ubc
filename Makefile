install:
	bash install.sh

update:
	pre-commit autoupdate --bleeding-edge

link:
	lygadgets_link lygadgets
	lygadgets_link tech/ubc

test:
	pytest -s

test-force:
	rm -r gds_ref
	pytest --force-regen

diff:
	pf merge-cells gds_diff

cov:
	pytest --cov=ubc

mypy:
	mypy . --ignore-missing-imports

lint:
	flake8 .

pylint:
	pylint ubc

lintd:
	flake8 --select RST

pydocstyle:
	pydocstyle ubc

doc8:
	doc8 docs/
