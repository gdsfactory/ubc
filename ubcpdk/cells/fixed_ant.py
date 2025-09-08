import gdsfactory as gf

from ubcpdk.config import PATH
from ubcpdk.import_gds import import_gds

gdsdir = PATH.gds_ant


@gf.cell
def ebeam_splitter_swg_assist_te1310_ANT() -> gf.Component:
    """Returns ebeam_splitter_swg_assist_te1310_ANT fixed cell."""
    return import_gds(gdsdir / "ebeam_splitter_swg_assist_te1310_ANT.GDS")


@gf.cell
def ebeam_splitter_swg_assist_te1550_ANT() -> gf.Component:
    """Returns ebeam_splitter_swg_assist_te1550_ANT fixed cell."""
    return import_gds(gdsdir / "ebeam_splitter_swg_assist_te1550_ANT.GDS")
