import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="ubc",
    version="0.0.2",
    url="https://github.com/gdsfactory/ubc",
    license="MIT",
    author="Joaquin",
    author_email="j",
    description="UBC Siepic Ebeam PDK from edx course",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ],
)
