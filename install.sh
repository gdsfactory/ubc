#!/bin/sh

pip install -e .
pip install pre-commit
pre-commit install
python install_tech.py
