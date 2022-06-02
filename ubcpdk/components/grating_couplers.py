import gdsfactory as gf
from ubcpdk.import_gds import add_ports_from_siepic_pins
from ubcpdk.config import PATH


decorator = gf.compose(
    gf.functions.rotate180,
    add_ports_from_siepic_pins,
)

import_gc = gf.partial(
    gf.import_gds,
    gdsdir=PATH.gds,
    library="Design kits/ebeam",
    decorator=decorator,
    layout_model_port_pairs=(("opt1", "opt_wg"),),
)

gc_te1550 = gf.partial(
    import_gc,
    gdspath="ebeam_gc_te1550.gds",
    polarization="te",
    wavelength=1.55,
    model="ebeam_gc_te1550",
    name="ebeam_gc_te1550",
)

gc_te1550_broadband = gf.partial(
    import_gc,
    gdspath="ebeam_gc_te1550_broadband.gds",
    name="ebeam_gc_te1550_broadband",
    model="ebeam_gc_te1550_broadband",
    polarization="te",
    wavelength=1.55,
)


gc_te1310 = gf.partial(
    import_gc,
    gdspath="ebeam_gc_te1310.gds",
    name="ebeam_gc_te1310",
    model="ebeam_gc_te1310",
    polarization="te",
    wavelength=1.31,
)

gc_tm1550 = gf.partial(
    import_gc,
    gdspath="ebeam_gc_tm1550.gds",
    name="ebeam_gc_tm1550",
    model="ebeam_gc_tm1550",
    polarization="tm",
    wavelength=1.55,
)


def test_gc():
    c1 = gc_tm1550()
    c2 = gc_te1550()
    c3 = gc_te1310()
    assert c1.name != c2.name != c3.name, f"{c1.name} {c2.name} {c3.name}"


if __name__ == "__main__":
    # from gdsfactory.serialization import clean_value_name
    # d = clean_value_name(decorator)
    # print(d)

    # test_gc()
    # c = gc_tm1550()
    c = gc_te1310()
    # c = gc_te1550()
    c.show()
