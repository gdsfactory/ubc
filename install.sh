#!/bin/sh

pip install -r requirements.txt --upgrade
pip install -r requirements_dev.txt --upgrade
pip install pre-commit
pre-commit install
python install_tech.py
