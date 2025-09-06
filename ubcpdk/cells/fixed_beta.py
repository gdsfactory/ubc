import gdsfactory as gf

from ubcpdk.config import PATH
from ubcpdk.import_gds import import_gds

gdsdir = PATH.gds_beta


@gf.cell
def Alignment_Marker() -> gf.Component:
    """Returns Alignment_Marker fixed cell."""
    return import_gds(gdsdir / "Alignment_Marker.GDS")


@gf.cell
def Packaging_FibreArray_8ch() -> gf.Component:
    """Returns Packaging_FibreArray_8ch fixed cell."""
    return import_gds(gdsdir / "Packaging_FibreArray_8ch.gds")


@gf.cell
def SEM_example() -> gf.Component:
    """Returns SEM_example fixed cell."""
    return import_gds(gdsdir / "SEM_example.gds")


@gf.cell
def ebeam_BondPad() -> gf.Component:
    """Returns ebeam_BondPad fixed cell."""
    return import_gds(gdsdir / "ebeam_BondPad.gds")


@gf.cell
def ebeam_BondPad_75() -> gf.Component:
    """Returns ebeam_BondPad_75 fixed cell."""
    return import_gds(gdsdir / "ebeam_BondPad_75.gds")


@gf.cell
def ebeam_bdc_tm1550() -> gf.Component:
    """Returns ebeam_bdc_tm1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_bdc_tm1550.gds")


@gf.cell
def ebeam_gc_te1310() -> gf.Component:
    """Returns ebeam_gc_te1310 fixed cell."""
    return import_gds(gdsdir / "ebeam_gc_te1310.gds")


@gf.cell
def ebeam_gc_te1310_broadband() -> gf.Component:
    """Returns ebeam_gc_te1310_broadband fixed cell."""
    return import_gds(gdsdir / "ebeam_gc_te1310_broadband.gds")


@gf.cell
def ebeam_gc_te1550_90nmSlab() -> gf.Component:
    """Returns ebeam_gc_te1550_90nmSlab fixed cell."""
    return import_gds(gdsdir / "ebeam_gc_te1550_90nmSlab.gds")


@gf.cell
def ebeam_gc_te1550_broadband() -> gf.Component:
    """Returns ebeam_gc_te1550_broadband fixed cell."""
    return import_gds(gdsdir / "ebeam_gc_te1550_broadband.GDS")


@gf.cell
def ebeam_splitter_adiabatic_swg_te1550() -> gf.Component:
    """Returns ebeam_splitter_adiabatic_swg_te1550 fixed cell."""
    return import_gds(gdsdir / "ebeam_splitter_adiabatic_swg_te1550.gds")


@gf.cell
def ebeam_swg_edgecoupler() -> gf.Component:
    """Returns ebeam_swg_edgecoupler fixed cell."""
    return import_gds(gdsdir / "ebeam_swg_edgecoupler.gds")


@gf.cell
def ebeam_terminator_te1310() -> gf.Component:
    """Returns ebeam_terminator_te1310 fixed cell."""
    return import_gds(gdsdir / "ebeam_terminator_te1310.gds")


@gf.cell
def ebeam_y_adiabatic_1310() -> gf.Component:
    """Returns ebeam_y_adiabatic_1310 fixed cell."""
    return import_gds(gdsdir / "ebeam_y_adiabatic_1310.gds")


@gf.cell
def metal_via() -> gf.Component:
    """Returns metal_via fixed cell."""
    return import_gds(gdsdir / "metal_via.gds")


@gf.cell
def pbs_1550_eskid() -> gf.Component:
    """Returns pbs_1550_eskid fixed cell."""
    return import_gds(gdsdir / "pbs_1550_eskid.gds")


@gf.cell
def photonic_wirebond_surfacetaper_1310() -> gf.Component:
    """Returns photonic_wirebond_surfacetaper_1310 fixed cell."""
    return import_gds(gdsdir / "photonic_wirebond_surfacetaper_1310.gds")


@gf.cell
def photonic_wirebond_surfacetaper_1550() -> gf.Component:
    """Returns photonic_wirebond_surfacetaper_1550 fixed cell."""
    return import_gds(gdsdir / "photonic_wirebond_surfacetaper_1550.gds")


@gf.cell
def siepic_o_gc_te1270_BB() -> gf.Component:
    """Returns siepic_o_gc_te1270_BB fixed cell."""
    return import_gds(gdsdir / "siepic_o_gc_te1270_BB.GDS")


@gf.cell
def siepic_o_pwbstlas_si_BB() -> gf.Component:
    """Returns siepic_o_pwbstlas_si_BB fixed cell."""
    return import_gds(gdsdir / "siepic_o_pwbstlas_si_BB.GDS")


@gf.cell
def thermal_phase_shifter_multimode_() -> gf.Component:
    """Returns thermal_phase_shifters fixed cell."""
    return import_gds(gdsdir / "thermal_phase_shifter_multimode_.gds")


@gf.cell
def thermal_phase_shifter_te_1310_() -> gf.Component:
    """Returns thermal_phase_shifters fixed cell."""
    return import_gds(gdsdir / "thermal_phase_shifter_te_1310_.gds")


@gf.cell
def thermal_phase_shifter_te_1310_50() -> gf.Component:
    """Returns thermal_phase_shifters fixed cell."""
    return import_gds(gdsdir / "thermal_phase_shifter_te_1310_50.gds")


@gf.cell
def thermal_phase_shifter_te_1550_50() -> gf.Component:
    """Returns thermal_phase_shifters fixed cell."""
    return import_gds(gdsdir / "thermal_phase_shifter_te_1550_50.gds")


if __name__ == "__main__":
    from ubcpdk import PDK

    PDK.activate()
    c = thermal_phase_shifter_te_1310_()
    # c = thermal_phase_shifter_multimode_()
    c.show()
