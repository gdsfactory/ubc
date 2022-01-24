# UBC PDK 0.0.10

UBC SiEPIC Ebeam PDK from [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)

## Installation

### Installation for users

You can install directly from pip `pip install ubcpdk` specify a specific version `pip install ubcpdk==0.0.10`
and update to the latest version with `pip install ubcpdk --upgrade`

If you are on Windows, I recommend you install gdsfactory with Anaconda3 or Miniconda3.

I also reccommend you install the gdsfactory link to klayout `gt tool install`

```
conda install -c conda-forge gdspy
pip install ubcpdk --upgrade
gf tool install
```

### Installation for developers

For developers you need to `git clone` the github repository, fork it, git add, git commit, git push and merge request your changes.

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

UBC pdk top contributors:

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
