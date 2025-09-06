import gdsfactory as gf

from ubcpdk import cells, PDK
from ubcpdk.tech import LAYER

size = (440, 470)
add_gc = cells.add_fiber_array


@gf.cell
def EBeam_JoaquinMatres_5() -> gf.Component:
    """Ring resonators."""

    rings = []
    for length_x in [4]:
        ring = cells.ring_single_heater(length_x=length_x)
        ring_gc = cells.add_fiber_array_pads_rf(ring)
        rings.append(ring_gc)

    c = gf.Component()
    _ = c << gf.pack(rings, max_size=size, spacing=2)[0]
    _ = c << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return c


def test_ubc5() -> None:
    c = EBeam_JoaquinMatres_5()
    assert c


if __name__ == "__main__":
    PDK.activate()
    c = EBeam_JoaquinMatres_5()
    c.show()  # show in klayout
