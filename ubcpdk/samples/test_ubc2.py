import gdsfactory as gf

import ubcpdk.components as uc
from ubcpdk.tech import LAYER

size = (440, 470)
add_gc = uc.add_fiber_array


@gf.cell
def EBeam_JoaquinMatres_2() -> gf.Component:
    """spirals for extracting straight waveguide loss"""
    e = [
        uc.add_fiber_array(component=uc.spiral(n_loops=8, length=length))
        for length in [0, 100, 200]
    ]

    c = gf.Component()
    _ = c << gf.pack(e, max_size=size, spacing=2)[0]
    _ = c << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return c


def test_ubc2() -> None:
    c = EBeam_JoaquinMatres_2()
    assert c


if __name__ == "__main__":
    c = EBeam_JoaquinMatres_2()
    c.show()  # show in klayout
