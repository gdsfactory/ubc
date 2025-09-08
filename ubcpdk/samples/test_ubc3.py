import gdsfactory as gf

from ubcpdk import PDK, cells


@gf.cell
def EBeam_JoaquinMatres_3() -> gf.Component:
    """Contains mirror cavities and structures inside a resonator."""
    size = (440, 470)
    add_gc = cells.add_fiber_array

    e = []
    e += [add_gc(cells.ebeam_crossing4())]
    e += [add_gc(cells.ebeam_adiabatic_te1550())]
    e += [add_gc(cells.ebeam_bdc_te1550())]
    e += [add_gc(cells.ebeam_y_1550())]
    e += [add_gc(cells.straight(), component_name=f"straight_{i}") for i in range(2)]
    c = gf.Component()
    _ = c << gf.pack(e, max_size=size, spacing=2)[0]
    _ = c << cells.die()
    return c


def test_ubc3() -> None:
    c = EBeam_JoaquinMatres_3()
    assert c


if __name__ == "__main__":
    PDK.activate()
    c = EBeam_JoaquinMatres_3()
    c.show()  # show in klayout
