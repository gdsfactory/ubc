# ubcpdk (SiEPIC Ebeam PDK) 1.21.0

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)
[![pypi](https://img.shields.io/pypi/v/ubcpdk)](https://pypi.org/project/ubcpdk/)
[![issues](https://img.shields.io/github/issues/gdsfactory/ubc)](https://github.com/gdsfactory/ubc/issues)
![forks](https://img.shields.io/github/forks/gdsfactory/ubc)
![Stars](https://img.shields.io/github/stars/gdsfactory/ubc)
[![mit](https://img.shields.io/github/license/gdsfactory/ubc)](https://choosealicense.com/licenses/mit/)
[![codecov](https://codecov.io/gh/gdsfactory/ubc/branch/main/graph/badge.svg?token=T3kCV2gYE9)](https://codecov.io/gh/gdsfactory/ubc)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

SiEPIC Ebeam PDK adapted from [siepic Ebeam PDK](https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK) for gdsfactory.
It provides a fully python driven flow alternative for the most advanced users taking the [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)

## Installation

### Installation for new python users

If you don't have python installed on your system you can [download the gdsfactory installer](https://github.com/gdsfactory/gdsfactory/releases) that includes python3, miniconda and all gdsfactory plugins.

### Installation for new gdsfactory users

If you already have python installed. Open Anaconda Prompt and then install the ubcpdk using pip.

![anaconda prompt](https://i.imgur.com/Fyal5sT.png)

```
pip install ubcpdk --upgrade
gf tool install
```

Then you need to restart Klayout to make sure the new technology installed appears.

### Installation for developers

For developers you need to `git clone` the GitHub repository, fork it, git add, git commit, git push and merge request your changes.

```
git clone https://github.com/gdsfactory/ubc.git
cd ubc
pip install -e . pre-commit
pre-commit install
python install_tech.py
gf tool install
```

## Documentation

- Run notebooks on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)
- [UBCpdk docs](https://gdsfactory.github.io/ubc/) and [code](https://github.com/gdsfactory/ubc)
- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)

## Acks

UBCpdk top contributors:

- Joaquin Matres (Google): maintainer of gdsfactory
- Thomas Dorch (Freedom Photonics): for Meep's material database access, MPB sidewall angles, and add_pin_path
- Lukas Chrostowski (UBC professor): creator of the course and maintainer of the PDK cells

Links:

- [UBC docs](https://gdsfactory.github.io/ubc/) and [repo](https://github.com/gdsfactory/ubc)
- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
- [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)
- [awesome photonics list](https://github.com/joamatab/awesome_photonics)
