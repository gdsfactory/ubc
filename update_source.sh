#!/bin/bash

export VERSION=0.3.41
export CML_VER=2022_02_01

export GIT_REPO=https://github.com/SiEPIC/SiEPIC_EBeam_PDK

export REPO_TECH=SiEPIC_EBeam_PDK-${VERSION}/klayout_dot_config/tech/EBeam
export UBC_TECH=ubcpdk/klayout/tech
export LUM_DIR=ubcpdk/simulation/lumerical
export GDS_DIR=ubcpdk/gds


# Download EBeam PDK archive and extract it here
wget ${GIT_REPO}/archive/refs/tags/v${VERSION}.tar.gz
tar -xvf v${VERSION}.tar.gz

# Copy files from tech
cp -r ${REPO_TECH}/klayout_Layers_EBeam.lyp ${UBC_TECH}/layers.lyp
cp -r ${REPO_TECH}/EBeam.lyt ${UBC_TECH}/tech.lyt
cp -r ${REPO_TECH}/EBeam_v${CML_VER}.cml ${LUM_DIR}/EBeam.cml
cp -r ${REPO_TECH}/lumerical_process_file.lbr ${LUM_DIR}/lumerical_process_file.lbr

# TODO: Make this work with the naming used by SiEPIC
# gf gds layermap_to_dataclass ${UBC_TECH}/layers.lyp

# Copy files from gds
cp -r ${REPO_TECH}/gds/development/*.gds ${GDS_DIR}
cp -r ${REPO_TECH}/gds/mature/*.gds ${GDS_DIR}

# Clean up
rm -r v${VERSION}.tar.gz SiEPIC_EBeam_PDK-${VERSION}
