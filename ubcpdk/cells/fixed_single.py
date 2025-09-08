import gdsfactory as gf

from ubcpdk.config import PATH
from ubcpdk.import_gds import import_gds

gdsdir = PATH.gds_single


@gf.cell
def ANT_MMI_1x2_te1550_3dB_BB() -> gf.Component:
    """Returns ANT_MMI_1x2_te1550_3dB_BB fixed cell."""
    return import_gds(gdsdir / "ANT_MMI_1x2_te1550_3dB_BB.gds")


@gf.cell
def GC_SiN_TE_1310_8degOxide_BB() -> gf.Component:
    """Returns GC_SiN_TE_1310_8degOxide_BB fixed cell."""
    return import_gds(gdsdir / "GC_SiN_TE_1310_8degOxide_BB.gds")


@gf.cell
def GC_SiN_TE_1550_8degOxide_BB() -> gf.Component:
    """Returns GC_SiN_TE_1550_8degOxide_BB fixed cell."""
    return import_gds(gdsdir / "GC_SiN_TE_1550_8degOxide_BB.gds")


@gf.cell
def ebeam_MMI_2x2_5050_te1310() -> gf.Component:
    """Returns ULaval fixed cell."""
    cell = "ebeam_MMI_2x2_5050_te1310.gds"
    return import_gds(gdsdir / cell)


@gf.cell
def ebeam_YBranch_te1310() -> gf.Component:
    """Returns ULaval fixed cell."""
    cell = "ebeam_YBranch_te1310.gds"
    return import_gds(gdsdir / cell)


@gf.cell
def crossing_SiN_1550() -> gf.Component:
    """Returns crossing_SiN_1550 fixed cell."""
    return import_gds(gdsdir / "crossing_SiN_1550.gds")


@gf.cell
def crossing_SiN_1550_extended() -> gf.Component:
    """Returns crossing_SiN_1550_extended fixed cell."""
    return import_gds(gdsdir / "crossing_SiN_1550_extended.gds")


@gf.cell
def crossing_horizontal() -> gf.Component:
    """Returns crossing_horizontal fixed cell."""
    return import_gds(gdsdir / "crossing_horizontal.gds")


@gf.cell
def crossing_manhattan() -> gf.Component:
    """Returns crossing_manhattan fixed cell."""
    return import_gds(gdsdir / "crossing_manhattan.gds")


@gf.cell
def ebeam_BondPad() -> gf.Component:
    """Returns ebeam_BondPad fixed cell."""
    return import_gds(gdsdir / "ebeam_BondPad.gds")


@gf.cell
def ebeam_DC_2m1_te895() -> gf.Component:
    """Returns ebeam_DC_2m1_te895 fixed cell."""
    return import_gds(gdsdir / "ebeam_DC_2-1_te895.gds")


@gf.cell
def ebeam_DC_te895() -> gf.Component:
    """Returns ebeam_DC_te895 fixed cell."""
    return import_gds(gdsdir / "ebeam_DC_te895.gds")


@gf.cell
def ebeam_Polarizer_TM_1550_UQAM() -> gf.Component:
    """Returns ebeam_Polarizer_TM_1550_UQAM fixed cell."""
    return import_gds(gdsdir / "ebeam_Polarizer_TM_1550_UQAM.gds")


@gf.cell
def ebeam_YBranch_895() -> gf.Component:
    """Returns ebeam_YBranch_895 fixed cell."""
    return import_gds(gdsdir / "ebeam_YBranch_895.gds")


@gf.cell
def ebeam_gc_te895() -> gf.Component:
    """Returns ebeam_gc_te895 fixed cell."""
    return import_gds(gdsdir / "ebeam_gc_te895.gds")


@gf.cell
def ebeam_terminator_SiN_1310() -> gf.Component:
    """Returns ebeam_terminator_SiN_1310 fixed cell."""
    return import_gds(gdsdir / "ebeam_terminator_SiN_1310.gds")


@gf.cell
def ebeam_terminator_SiN_1550() -> gf.Component:
    """Returns ebeam_terminator_SiN_1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_terminator_SiN_1550.gds")


@gf.cell
def ebeam_terminator_SiN_te895() -> gf.Component:
    """Returns ebeam_terminator_SiN_te895 fixed cell."""
    return import_gds(gdsdir / "ebeam_terminator_SiN_te895.gds")


@gf.cell
def taper_SiN_750_3000() -> gf.Component:
    """Returns taper_SiN_750_3000 fixed cell."""
    return import_gds(gdsdir / "taper_SiN_750_3000.gds")


if __name__ == "__main__":
    from ubcpdk import PDK

    PDK.activate()
    c = ebeam_YBranch_te1310()
    # gdspath = c.write_gds()
    # gf.show(gdspath)
    c.pprint_ports()
    c.show()
