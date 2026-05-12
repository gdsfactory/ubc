# ubcpdk (SiEPIC Ebeam PDK) 3.3.4

<!-- BADGES:START -->
[![Docs](https://github.com/gdsfactory/ubc/actions/workflows/pages.yml/badge.svg)](https://github.com/gdsfactory/ubc/actions/workflows/pages.yml)
[![Tests](https://github.com/gdsfactory/ubc/actions/workflows/test_code.yml/badge.svg)](https://github.com/gdsfactory/ubc/actions/workflows/test_code.yml)
[![DRC](https://github.com/gdsfactory/ubc/raw/badges/drc.svg)](https://github.com/gdsfactory/ubc/actions/workflows/drc.yml)
[![Model Regression](https://github.com/gdsfactory/ubc/actions/workflows/model_regression.yml/badge.svg)](https://github.com/gdsfactory/ubc/actions/workflows/model_regression.yml)
[![Test Coverage](https://github.com/gdsfactory/ubc/raw/badges/coverage.svg)](https://github.com/gdsfactory/ubc/actions/workflows/test_coverage.yml)
[![Model Coverage](https://github.com/gdsfactory/ubc/raw/badges/model_coverage.svg)](https://github.com/gdsfactory/ubc/actions/workflows/model_coverage.yml)
[![Issues](https://github.com/gdsfactory/ubc/raw/badges/issues.svg)](https://github.com/gdsfactory/ubc/issues)
[![PRs](https://github.com/gdsfactory/ubc/raw/badges/prs.svg)](https://github.com/gdsfactory/ubc/pulls)
<!-- BADGES:END -->


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)
[![pypi](https://img.shields.io/pypi/v/ubcpdk)](https://pypi.org/project/ubcpdk/)
[![mit](https://img.shields.io/github/license/gdsfactory/ubc)](https://choosealicense.com/licenses/mit/)
[![codecov](https://codecov.io/gh/gdsfactory/ubc/branch/main/graph/badge.svg?token=T3kCV2gYE9)](https://codecov.io/gh/gdsfactory/ubc)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

SiEPIC Ebeam PDK adapted from [siepic Ebeam PDK](https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK) for gdsfactory.
It provides a fully python driven flow alternative for the most advanced users taking the [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)

## Installation

We recommend `uv`

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation for users

Use python 3.11, 3.12 or 3.13. We recommend [VSCode](https://code.visualstudio.com/) as an IDE.

```
uv pip install ubcpdk --upgrade
```

Then you need to restart Klayout to make sure the new technology installed appears.

### Installation for contributors


Then you can install with:

```bash
git clone https://github.com/gdsfactory/ubc.git
cd ubc
uv venv --python 3.12
uv sync --extra docs --extra dev
```

## Documentation

- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
- [UBCpdk docs](https://gdsfactory.github.io/ubc/) and [code](https://github.com/gdsfactory/ubc)
- [ubc1](https://github.com/gdsfactory/ubc1)

## Pre-commit

```bash
make pre-commit
```

## Release

1. Bump the version:

```bash
tbump 0.0.1
```

2. Push the tag:

```bash
git push --tags
```
This triggers the release workflow that builds wheels and uploads them.

3. Create a pull request with the updated changelog since last release.
