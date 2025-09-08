import gdsfactory as gf

from ubcpdk import PDK, cells


@gf.cell
def EBeam_JoaquinMatres_5() -> gf.Component:
    """Ring resonators."""
    size = (440, 470)

    rings = []
    for length_x in [4]:
        ring = cells.ring_single_heater(length_x=length_x)
        ring_gc = cells.add_fiber_array_pads_rf(ring)
        rings.append(ring_gc)

    c = gf.Component()
    _ = c << gf.pack(rings, max_size=size, spacing=2)[0]
    _ = c << cells.die()
    return c


def test_ubc5() -> None:
    c = EBeam_JoaquinMatres_5()
    assert c


if __name__ == "__main__":
    PDK.activate()
    c = EBeam_JoaquinMatres_5()
    c.show()  # show in klayout
