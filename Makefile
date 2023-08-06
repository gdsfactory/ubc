install:
	pip install -e .
	pip install pre-commit
	pre-commit install
	python install_tech.py

dev:
	pip install -e .[dev,docs]

update-pre:
	pre-commit autoupdate

watch:
	gf watch ubcpdk

link:
	lygadgets_link lygadgets
	lygadgets_link gdsfactory
	lygadgets_link toolz
	lygadgets_link gdspy
	lygadgets_link numpy
	lygadgets_link matplotlib
	lygadgets_link cycler
	lygadgets_link pyparsing
	lygadgets_link dateutil
	lygadgets_link kiwisolver
	lygadgets_link ubcpdk/klayout/tech
	lygadgets_link ubcpdk
	lygadgets_link scipy  # [because of splines in the nanowires]
	lygadgets_link omegaconf
	lygadgets_link loguru
	lygadgets_link pydantic
	lygadgets_link shapely

test:
	pytest -s

test-force:
	rm -r gds_ref
	pytest --force-regen

doc:
	python docs/write_components_autodoc.py

meep:
	conda install -n base conda-libmamba-solver
	conda config --set solver libmamba
	conda install -c conda-forge pymeep=*=mpi_mpich_* nlopt -y

plugins: meep
	pip install gdsfactory[docs,dev,full,gmsh,tidy3d,devsim,meow,sax]
	pip install -e .[full] --upgrade

diff:
	pf merge-cells gds_diff

cov:
	pytest --cov=ubcpdk

mypy:
	mypy . --ignore-missing-imports

lint:
	flake8 .

pylint:
	pylint ubcpdk

lintd:
	flake8 --select RST

pydocstyle:
	pydocstyle ubcpdk

doc8:
	doc8 docs/

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

release:
	git push
	git push origin --tags

build:
	rm -rf dist
	pip install build
	python -m build

notebooks:
	nbstripout --drop-empty-cells docs/notebooks/*.ipynb

mask:
	python ubcpdk/samples/test_masks.py

jupytext:
	jupytext docs/**/*.ipynb --to py

notebooks:
	jupytext docs/**/*.py --to ipynb

notebooks-rm:
	rm docs/notebooks/*.ipynb

docs:
	jb build docs

.PHONY: drc doc docs
