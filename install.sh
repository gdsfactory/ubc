#!/bin/sh

pip install pre-commit flake8
poetry install
pre-commit install
python install_tech.py
