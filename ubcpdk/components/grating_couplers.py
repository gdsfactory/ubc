import gdsfactory as gf
from ubcpdk.import_gds import add_ports_renamed_gratings, add_ports_siepic_gratings, import_gds_siepic_pins


# This rotation is causing issues in interconnect
add_ports_rotate180 = gf.compose(gf.functions.rotate180, add_ports_renamed_gratings)
add_ports_rotate180_siepic = gf.compose(gf.functions.rotate180, add_ports_siepic_gratings)

import_gc = gf.partial(
    import_gds_siepic_pins,
    decorator=add_ports_rotate180,
)

import_gc_interconnect = gf.partial(
    import_gds_siepic_pins,
    decorator=add_ports_rotate180_siepic,
)

gc_te1550 = gf.partial(
    import_gc_interconnect,
    "ebeam_gc_te1550.gds",
    polarization="te",
    wavelength=1.55,
    model="ebeam_gc_te1550",
    opt1="opt_wg",
)

gc_te1550_broadband = gf.partial(
    import_gc,
    "ebeam_gc_te1550_broadband.gds",
    polarization="te",
    wavelength=1.55,
)


gc_te1310 = gf.partial(
    import_gc,
    "ebeam_gc_te1310.gds",
    polarization="te",
    wavelength=1.31,
)

gc_tm1550 = gf.partial(
    import_gc,
    "ebeam_gc_tm1550.gds",
    polarization="tm",
    wavelength=1.55,
)


if __name__ == "__main__":
    # c = gc_te1310()
    # c = gc_tm1550()
    c = gc_te1550()
    c.show()
