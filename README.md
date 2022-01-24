# UBCpdk 0.0.12

[![](https://img.shields.io/pypi/v/ubcpdk)](https://pypi.org/project/ubcpdk/)
[![](https://img.shields.io/github/issues/gdsfactory/ubc)](https://github.com/gdsfactory/ubc/issues)
![](https://img.shields.io/github/forks/gdsfactory/ubc)
![](https://img.shields.io/github/stars/gdsfactory/ubc)
[![](https://img.shields.io/github/license/gdsfactory/ubc)](https://choosealicense.com/licenses/mit/)
[![](https://img.shields.io/codecov/c/github/gdsfactory/ubc)](https://codecov.io/gh/gdsfactory/ubc/tree/master/ubc)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


UBC SiEPIC Ebeam PDK from [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)

## Installation

### Installation for users

You can install directly from pip `pip install ubcpdk` specify a specific version `pip install ubcpdk==0.0.12`
and update to the latest version with `pip install ubcpdk --upgrade`

If you are on Windows, I recommend you install gdsfactory with Mamba (faster conda) and Pip.

I also recommend you install the gdsfactory link to klayout `gt tool install`

```
conda install -c conda-forge gdspy
pip install ubcpdk --upgrade
gf tool install
```

If you want to get all the extras (holoviews plots, tensorflow modes, modesolver ...)

```
conda install -c conda-forge gdspy
pip install ubcpdk[full] --upgrade
gf tool install
```

### Installation for developers

For developers you need to `git clone` the GitHub repository, fork it, git add, git commit, git push and merge request your changes.

```
git clone https://github.com/gdsfactory/ubc.git
cd ubc
pip install -r requirements.txt --upgrade
pip install -r requirements_dev.txt --upgrade
pip install pre-commit
pre-commit install
python install_tech.py
gf tool install
```

## Acks

UBCpdk top contributors:

- Lukas Chrostowski (UBC professor): creator of the course and maintainer of the PDK cells
- Joaquin Matres (Google): maintainer of gdsfactory
- Alex Tait (Queens University): maintainer of lygadgets

Open source heroes:

- Matthias Köfferlein (Germany): for Klayout
- Lucas Heitzmann (University of Campinas, Brazil): for gdspy
- Adam McCaughan (NIST): for phidl
- Alex Tait (Queens University): for lytest
- Thomas Ferreira de Lima (NEC): for `pip install klayout`


Links:

- [UBC docs](https://gdsfactory.github.io/ubc/) and [repo](https://github.com/gdsfactory/ubc)
- [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)
- [siepic Ebeam PDK](https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK)
- [gdsfactory](https://gdsfactory.github.io/gdsfactory/)
- [awesome photonics list](https://github.com/joamatab/awesome_photonics)
