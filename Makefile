install:
	bash install.sh

update:
	pre-commit autoupdate --bleeding-edge

watch:
	gf yaml watch ubcpdk

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
	mamba install pymeep=*=mpi_mpich_* -y

sax:
	pip install sax jax jaxlib

plugins: sax meep

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

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

release:
	git push
	git push origin --tags
