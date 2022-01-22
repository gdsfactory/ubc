# UBC PDK 0.0.6

UBC SiEPIC Ebeam PDK from [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)

## Installation for users

You can install directly from pip `pip install ubc`

pip also lets you install a specific version `pip install ubc==0.0.7`

and update to the latest version with `pip install ubc --upgrade`

## Installation for developers

Run `make install` in a terminal. If you are on Windows, open an anaconda prompt terminal and type:

```
git clone https://github.com/gdsfactory/ubc.git
cd ubc
pip install -r requirements.txt --upgrade
pip install -r requirements_dev.txt --upgrade
pip install pre-commit
pre-commit install
python install_tech.py
```

## Acks

- [edx course](https://www.edx.org/course/silicon-photonics-design-fabrication-and-data-ana)
- [siepic Ebeam PDK](https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK)
- [gdsfactory](https://gdsfactory.github.io/gdsfactory/)

Main developers:

- Lukas Chrostowski: creator of the course and maintainer of the PDK cells
- Joaquin Matres: maintainer of gdsfactory
- Alex Tait: maintainer of lygadgets
