import gdsfactory as gf
from ubcpdk.import_gds import add_ports_renamed_gratings, import_gds_siepic_pins

# This rotation is causing issues in interconnect
add_ports_rotate180 = gf.compose(gf.functions.rotate180, add_ports_renamed_gratings)


gc_te1550 = gf.partial(
    import_gds_siepic_pins,
    "ebeam_gc_te1550.gds",
    decorator=gf.functions.rotate180,
    polarization="te",
    wavelength=1.55,
    name="ebeam_gc_te1550",
    opt1="opt_wg",
)

gc_te1550_broadband = gf.partial(
    import_gds_siepic_pins,
    "ebeam_gc_te1550_broadband.gds",
    decorator=gf.functions.rotate180,
    polarization="te",
    wavelength=1.55,
)


gc_te1310 = gf.partial(
    import_gds_siepic_pins,
    "ebeam_gc_te1310.gds",
    decorator=gf.functions.rotate180,
    polarization="te",
    wavelength=1.31,
)

gc_tm1550 = gf.partial(
    import_gds_siepic_pins,
    "ebeam_gc_tm1550.gds",
    decorator=gf.functions.rotate180,
    polarization="tm",
    wavelength=1.55,
)


if __name__ == "__main__":
    # c = gc_te1310()
    # c = gc_tm1550()
    c = gc_te1550()
    c.show()
