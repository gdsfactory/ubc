# ubcpdk (SiEPIC Ebeam PDK) 2.6.2

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)
[![pypi](https://img.shields.io/pypi/v/ubcpdk)](https://pypi.org/project/ubcpdk/)
[![mit](https://img.shields.io/github/license/gdsfactory/ubc)](https://choosealicense.com/licenses/mit/)
[![codecov](https://codecov.io/gh/gdsfactory/ubc/branch/main/graph/badge.svg?token=T3kCV2gYE9)](https://codecov.io/gh/gdsfactory/ubc)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

SiEPIC Ebeam PDK adapted from [siepic Ebeam PDK](https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK) for gdsfactory.
It provides a fully python driven flow alternative for the most advanced users taking the [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)

## Installation

### Installation for users

Use python3.10 or python3.11. We recommend [VSCode](https://code.visualstudio.com/) as an IDE.

If you don't have python installed on your system you can [download anaconda](https://www.anaconda.com/download/)

Once you have python installed, open Anaconda Prompt as Administrator and then install the latest gdsfactory using pip.

![anaconda prompt](https://i.imgur.com/eKk2bbs.png)
```
pip install ubcpdk --upgrade
```

Then you need to restart Klayout to make sure the new technology installed appears.

### Installation for developers

For developers you need to fork, `git clone` the GitHub repository, git add, git commit, git push and merge request your changes.

```
git clone https://github.com/gdsfactory/ubc.git
cd ubc
pip install -e . pre-commit
pre-commit install
python install_tech.py
```

## Documentation

- [UBCpdk docs](https://gdsfactory.github.io/ubc/) and [code](https://github.com/gdsfactory/ubc)
- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
- [ubc1](https://github.com/gdsfactory/ubc1)
