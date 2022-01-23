#!/bin/sh

poetry install
pip install pre-commit
pre-commit install
python install_tech.py
