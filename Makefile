install:
	pip install -e .[dev,docs]
	pip install pre-commit
	pre-commit install
	python install_tech.py

dev: install

update-pre:
	pre-commit autoupdate

tech:
	python install_tech.py

test:
	pytest -s

cov:
	pytest --cov=ubcpdk

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

meep:
	conda install -n base conda-libmamba-solver
	conda config --set solver libmamba
	conda install -c conda-forge pymeep=*=mpi_mpich_* nlopt -y

release:
	git push
	git push origin --tags

build:
	rm -rf dist
	pip install build
	python -m build

docs:
	jb build docs

mask:
	python ubcpdk/samples/test_masks.py

.PHONY: drc doc docs install
