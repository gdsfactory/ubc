import gdsfactory as gf
from ubcpdk.import_gds import import_gds


import_gc = gf.partial(import_gds, rotation=180)


gc_te1550 = gf.partial(
    import_gc,
    "ebeam_gc_te1550.gds",
    polarization="te",
    wavelength=1.55,
    model="ebeam_gc_te1550",
    name="ebeam_gc_te1550",
    layout_model_port_pairs=(("opt1", "opt_wg"),),
)

gc_te1550_broadband = gf.partial(
    import_gc,
    "ebeam_gc_te1550_broadband.gds",
    name="ebeam_gc_te1550_broadband",
    model="ebeam_gc_te1550_broadband",
    polarization="te",
    wavelength=1.55,
    layout_model_port_pairs=(("opt1", "opt_wg"),),
)


gc_te1310 = gf.partial(
    import_gc,
    "ebeam_gc_te1310.gds",
    name="ebeam_gc_te1310",
    model="ebeam_gc_te1310",
    polarization="te",
    wavelength=1.31,
    layout_model_port_pairs=(("opt1", "opt_wg"),),
)

gc_tm1550 = gf.partial(
    import_gc,
    "ebeam_gc_tm1550.gds",
    name="ebeam_gc_tm1550",
    model="ebeam_gc_tm1550",
    polarization="tm",
    wavelength=1.55,
    layout_model_port_pairs=(("opt1", "opt_wg"),),
)


if __name__ == "__main__":
    # c = gc_te1310()
    # c = gc_tm1550()
    c = gc_te1550()
    c.show()
