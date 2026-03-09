install:
	uv sync --extra docs --extra dev

all:
	uv run python ubcpdk/samples/all_cells.py

rm-samples:
	rm -rf ubcpdk/samples

dev: install

update-pre:
	pre-commit autoupdate

tech:
	python install_tech.py

test:
	uv run pytest -s

test-ports:
	uv run pytest -s tests/test_si220_cband.py::test_optical_port_positions

test-force: install
	uv run pytest -s --update-gds-refs --force-regen

cov:
	uv run pytest --cov=ubcpdk

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
	uv run python .github/write_components_plot.py
	uv run python .github/write_components_autodoc.py
	uv run jb build docs

mask:
	python ubcpdk/samples/test_masks.py

.PHONY: drc doc docs install
