import gdsfactory as gf

from ubcpdk.config import PATH
from ubcpdk.import_gds import import_gds

gdsdir = PATH.gds_dream


@gf.cell
def ebeam_dream_FAVE_SiN_1310_BB() -> gf.Component:
    """Returns ebeam_dream_FAVE_SiN_1310_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_dream_FAVE_SiN_1310_BB.gds")


@gf.cell
def ebeam_dream_FAVE_SiN_1550_BB() -> gf.Component:
    """Returns ebeam_dream_FAVE_SiN_1550_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_dream_FAVE_SiN_1550_BB.gds")


@gf.cell
def ebeam_dream_FAVE_Si_1310_BB() -> gf.Component:
    """Returns ebeam_dream_FAVE_Si_1310_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_dream_FAVE_Si_1310_BB.gds")


@gf.cell
def ebeam_dream_FAVE_Si_1550_BB() -> gf.Component:
    """Returns ebeam_dream_FAVE_Si_1550_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_dream_FAVE_Si_1550_BB.gds")


@gf.cell
def ebeam_dream_FaML_SiN_1550_BB() -> gf.Component:
    """Returns ebeam_dream_FaML_SiN_1550_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_dream_FaML_SiN_1550_BB.gds")


@gf.cell
def ebeam_dream_FaML_Si_1310_BB() -> gf.Component:
    """Returns ebeam_dream_FaML_Si_1310_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_dream_FaML_Si_1310_BB.gds")


@gf.cell
def ebeam_dream_FaML_Si_1550_BB() -> gf.Component:
    """Returns ebeam_dream_FaML_Si_1550_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_dream_FaML_Si_1550_BB.gds")


@gf.cell
def ebeam_dream_splitter_1x2_te1550_BB() -> gf.Component:
    """Returns ebeam_dream_splitter_1x2_te1550_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_dream_splitter_1x2_te1550_BB.gds")


@gf.cell
def ebeam_sin_dream_splitter1x2_te1550_BB() -> gf.Component:
    """Returns ebeam_sin_dream_splitter1x2_te1550_BB fixed cell."""
    return import_gds(gdsdir / "ebeam_sin_dream_splitter1x2_te1550_BB.gds")
