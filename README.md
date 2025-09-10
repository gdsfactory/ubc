# ubcpdk (SiEPIC Ebeam PDK) 3.2.0

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
uv pip install cspdk --upgrade
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
