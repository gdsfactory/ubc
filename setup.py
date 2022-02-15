import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


def get_install_requires():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("-")]


setup(
    name="ubcpdk",
    version="1.3.0",
    url="https://github.com/gdsfactory/ubc",
    include_package_data=True,
    license="MIT",
    author="gdsfactory",
    install_requires=get_install_requires(),
    description="UBC Siepic Ebeam PDK from edx course",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
