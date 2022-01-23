#!/bin/sh

poetry install
pip install pre-commit flake8
pre-commit install
python install_tech.py
