import gdsfactory as gf

from ubcpdk.config import PATH
from ubcpdk.import_gds import import_gds

gdsdir = PATH.gds_ebeam


@gf.cell
def GC_TE_1310_8degOxide_BB() -> gf.Component:
    """Returns GCs_BB fixed cell."""
    return import_gds(gdsdir / "GC_TE_1310_8degOxide_BB.gds")


@gf.cell
def GC_TE_1550_8degOxide_BB() -> gf.Component:
    """Returns GCs_BB fixed cell."""
    return import_gds(gdsdir / "GC_TE_1550_8degOxide_BB.gds")


@gf.cell
def GC_TM_1310_8degOxide_BB() -> gf.Component:
    """Returns GCs_BB fixed cell."""
    return import_gds(gdsdir / "GC_TM_1310_8degOxide_BB.gds")


@gf.cell
def GC_TM_1550_8degOxide_BB() -> gf.Component:
    """Returns GCs_BB fixed cell."""
    return import_gds(gdsdir / "GC_TM_1550_8degOxide_BB.gds")


@gf.cell
def ebeam_adiabatic_te1550() -> gf.Component:
    """Returns ebeam_adiabatic_te1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_adiabatic_te1550.gds")


@gf.cell
def ebeam_adiabatic_tm1550() -> gf.Component:
    """Returns ebeam_adiabatic_tm1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_adiabatic_tm1550.gds")


@gf.cell
def ebeam_bdc_te1550() -> gf.Component:
    """Returns ebeam_bdc_te1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_bdc_te1550.gds")


@gf.cell
def ebeam_crossing4() -> gf.Component:
    """Returns ebeam_crossing4 fixed cell."""
    return import_gds(gdsdir / "ebeam_crossing4.gds")


@gf.cell
def ebeam_gc_te1550() -> gf.Component:
    """Returns ebeam_gc_te1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_gc_te1550.gds")


@gf.cell
def ebeam_gc_tm1550() -> gf.Component:
    """Returns ebeam_gc_tm1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_gc_tm1550.gds")


@gf.cell
def ebeam_routing_taper_te1550_w500nm_to_w3000nm_L20um() -> gf.Component:
    """Returns ebeam_routing_taper_te1550_w500nm_to_w3000nm_L20um fixed cell."""
    return import_gds(
        gdsdir / "ebeam_routing_taper_te1550_w=500nm_to_w=3000nm_L=20um.gds"
    )


@gf.cell
def ebeam_routing_taper_te1550_w500nm_to_w3000nm_L40um() -> gf.Component:
    """Returns ebeam_routing_taper_te1550_w500nm_to_w3000nm_L40um fixed cell."""
    return import_gds(
        gdsdir / "ebeam_routing_taper_te1550_w=500nm_to_w=3000nm_L=40um.gds"
    )


@gf.cell
def ebeam_splitter_swg_assist_te1310() -> gf.Component:
    """Returns ebeam_splitter_swg_assist_te1310 fixed cell."""
    return import_gds(gdsdir / "ebeam_splitter_swg_assist_te1310.gds")


@gf.cell
def ebeam_splitter_swg_assist_te1550() -> gf.Component:
    """Returns ebeam_splitter_swg_assist_te1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_splitter_swg_assist_te1550.gds")


@gf.cell
def ebeam_terminator_te1310() -> gf.Component:
    """Returns ebeam_terminator_te1310 fixed cell."""
    return import_gds(gdsdir / "ebeam_terminator_te1310.gds")


@gf.cell
def ebeam_terminator_te1550() -> gf.Component:
    """Returns ebeam_terminator_te1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_terminator_te1550.gds")


@gf.cell
def ebeam_terminator_tm1550() -> gf.Component:
    """Returns ebeam_terminator_tm1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_terminator_tm1550.gds")


@gf.cell
def ebeam_y_1310() -> gf.Component:
    """Returns ebeam_y_1310 fixed cell."""
    return import_gds(gdsdir / "ebeam_y_1310.gds")


@gf.cell
def ebeam_y_1550() -> gf.Component:
    """Returns ebeam_y_1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_y_1550.gds")


@gf.cell
def ebeam_y_adiabatic() -> gf.Component:
    """Returns ebeam_y_adiabatic fixed cell."""
    return import_gds(gdsdir / "ebeam_y_adiabatic.gds")


@gf.cell
def ebeam_y_adiabatic_500pin() -> gf.Component:
    """Returns ebeam_y_adiabatic_500pin fixed cell."""
    return import_gds(gdsdir / "ebeam_y_adiabatic_500pin.gds")


@gf.cell
def taper_si_simm_1310() -> gf.Component:
    """Returns taper_si_simm_1310 fixed cell."""
    return import_gds(gdsdir / "taper_si_simm_1310.gds")


@gf.cell
def taper_si_simm_1550() -> gf.Component:
    """Returns taper_si_simm_1550 fixed cell."""
    return import_gds(gdsdir / "taper_si_simm_1550.gds")


if __name__ == "__main__":
    from ubcpdk import PDK

    PDK.activate()
    c = ebeam_y_1550()
    c.show()
