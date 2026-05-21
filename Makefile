install:
	uv sync --extra docs --extra dev

all:
	uv run python ubcpdk/samples/all_cells.py

rm-samples:
	rm -rf ubcpdk/samples

dev: install
	curl -sf https://raw.githubusercontent.com/doplaydo/pdk-ci-workflow/main/templates/.pre-commit-config.yaml -o .pre-commit-config.yaml
	uv run pre-commit install

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

mask:
	python ubcpdk/samples/test_masks.py

docs-pdf:
	uv run python .github/write_components_autodoc.py
	uv run python .github/write_components_plot.py
	uv run python -c "import re; from pathlib import Path; t=Path('CHANGELOG.md').read_text(); Path('docs/changelog.md').write_text(re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t))"
	uv run mkdocs build -f mkdocs-pdf.yml

docs:
	uv run python .github/write_components_autodoc.py
	uv run python .github/write_components_plot.py
	uv run python -c "import re; from pathlib import Path; t=Path('CHANGELOG.md').read_text(); Path('docs/changelog.md').write_text(re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t))"
	uv run --extra docs zensical build

docs-serve:
	uv run python .github/write_components_autodoc.py
	uv run python .github/write_components_plot.py
	uv run python -c "import re; from pathlib import Path; t=Path('CHANGELOG.md').read_text(); Path('docs/changelog.md').write_text(re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t))"
	uv run --extra docs zensical serve -a localhost:8080

.PHONY: drc drc-sample doc docs docs-pdf build
