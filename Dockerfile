FROM jupyter/base-notebook:python-3.8.8

# expose klive and jupyter notebook ports
EXPOSE 8082
EXPOSE 8083
EXPOSE 8888

USER root
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    # Common useful utilities
    git \
    neovim

USER jovyan
COPY . /home/jovyan/ubc
COPY docs/notebooks /home/jovyan/notebooks
RUN conda init bash

RUN mamba install gdspy -y
RUN mamba install pymeep=*=mpi_mpich_* -y

RUN pip install gdsfactory[full] triangle
WORKDIR /home/jovyan
