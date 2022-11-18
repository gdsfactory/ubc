from setuptools import find_packages, setup


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]


with open("requirements_simulations.txt") as f:
    requirements_full = [
        line.strip() for line in f.readlines() if not line.strip().startswith("-")
    ]


setup(
    name="ubcpdk",
    version="1.9.1",
    url="https://github.com/gdsfactory/ubc",
    include_package_data=True,
    license="MIT",
    author="gdsfactory",
    install_requires=requirements,
    # install_requires=('gdsfactory', 'modes', 'lygadgets')
    description="UBC Siepic Ebeam PDK from edx course",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    extras_require={
        "full": list(set(requirements + requirements_full)),
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
