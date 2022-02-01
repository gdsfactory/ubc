import gdsfactory as gf
from ubcpdk.import_gds import import_gds, add_ports_renamed


add_ports_rotate180 = gf.compose(gf.functions.rotate180, add_ports_renamed)

gc_te1550 = gf.partial(
    import_gds,
    "ebeam_gc_te1550.gds",
    decorator=add_ports_rotate180,
    polarization="te",
    wavelength=1.55,
)

gc_te1550_broadband = gf.partial(
    import_gds,
    "ebeam_gc_te1550_broadband.gds",
    decorator=add_ports_rotate180,
    polarization="te",
    wavelength=1.55,
)


gc_te1310 = gf.partial(
    import_gds,
    "ebeam_gc_te1310.gds",
    decorator=add_ports_rotate180,
    polarization="te",
    wavelength=1.31,
)

gc_tm1550 = gf.partial(
    import_gds,
    "ebeam_gc_tm1550.gds",
    decorator=add_ports_rotate180,
    polarization="tm",
    wavelength=1.55,
)


if __name__ == "__main__":
    c = gc_te1550()
    c.show()
