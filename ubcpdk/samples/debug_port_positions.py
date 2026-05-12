"""Debug cells with port position issues in library."""

import gdsfactory as gf

from ubcpdk import PDK

if __name__ == "__main__":
    PDK.activate()
    cell_name = "add_fiber_array_pads_rf"
    cell_name = "ANT_MMI_1x2_te1550_3dB_BB"
    cell_name = "crossing_horizontal"
    cell_name = "crossing_manhattan"
    cell_name = "crossing_SiN_1550"
    cell_name = "crossing_SiN_1550_extended"
    cell_name = "ebeam_DC_2m1_te895"
    cell_name = "ebeam_DC_te895"
    cell_name = "ebeam_dream_FaML_Si_1310_BB"
    cell_name = "ebeam_dream_FaML_Si_1550_BB"
    cell_name = "ebeam_dream_FaML_SiN_1550_BB"
    cell_name = "ebeam_dream_FAVE_Si_1310_BB"
    cell_name = "ebeam_dream_FAVE_Si_1550_BB"
    cell_name = "ebeam_dream_FAVE_SiN_1310_BB"
    cell_name = "ebeam_dream_FAVE_SiN_1550_BB"
    cell_name = "ebeam_gc_te1550"
    cell_name = "ebeam_gc_te1550_90nmSlab"
    cell_name = "ebeam_gc_te1550_broadband"
    cell_name = "ebeam_gc_te895"
    cell_name = "ebeam_gc_tm1550"
    cell_name = "ebeam_MMI_2x2_5050_te1310"
    cell_name = "ebeam_sin_dream_splitter1x2_te1550_BB"
    cell_name = "ebeam_terminator_SiN_1310"
    cell_name = "ebeam_terminator_SiN_1550"
    cell_name = "ebeam_terminator_SiN_te895"
    cell_name = "ebeam_YBranch_895"
    cell_name = "ebeam_YBranch_te1310"
    cell_name = "GC_SiN_TE_1310_8degOxide_BB"
    cell_name = "GC_SiN_TE_1550_8degOxide_BB"
    cell_name = "taper_SiN_750_3000"
    c = gf.get_component(cell_name)
    c.show()
