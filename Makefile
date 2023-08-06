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

cov:
	pytest --cov=ubcpdk

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

release:
	git push
	git push origin --tags


docs:
	jb build docs

.PHONY: drc doc docs
