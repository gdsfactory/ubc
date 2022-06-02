from setuptools import find_packages, setup


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


def get_install_requires():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("-")]


setup(
    name="ubcpdk",
    version="1.6.0",
    url="https://github.com/gdsfactory/ubc",
    include_package_data=True,
    license="MIT",
    author="gdsfactory",
    install_requires=get_install_requires(),
    # install_requires=('gdsfactory', 'modes', 'lygadgets')
    description="UBC Siepic Ebeam PDK from edx course",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
