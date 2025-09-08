import gdsfactory as gf

from ubcpdk import PDK, cells


@gf.cell
def EBeam_JoaquinMatres_1() -> gf.Component:
    """Add DBR cavities."""
    size = (440, 470)
    add_gc = cells.add_fiber_array

    e = [add_gc(cells.straight())]
    e += [add_gc(cells.mzi(delta_length=dl)) for dl in [9.32, 93.19]]
    e += [
        add_gc(cells.ring_single(radius=12, gap=gap, length_x=coupling_length))
        for gap in [0.2]
        for coupling_length in [2.5, 4.5, 6.5]
    ]

    c = gf.Component()
    _ = c << gf.pack(e, max_size=size, spacing=2)[0]
    _ = c << cells.die()
    return c


def test_ubc1() -> None:
    c = EBeam_JoaquinMatres_1()
    assert c


if __name__ == "__main__":
    PDK.activate()
    c = EBeam_JoaquinMatres_1()
    c.show()  # show in klayout
