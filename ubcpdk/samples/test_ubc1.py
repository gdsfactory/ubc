import gdsfactory as gf

import ubcpdk.components as uc
from ubcpdk.tech import LAYER

size = (440, 470)
add_gc = uc.add_fiber_array


@gf.cell
def EBeam_JoaquinMatres_1() -> gf.Component:
    """Add DBR cavities."""
    e = [add_gc(uc.straight())]
    e += [add_gc(uc.mzi(delta_length=dl)) for dl in [9.32, 93.19]]
    e += [
        add_gc(uc.ring_single(radius=12, gap=gap, length_x=coupling_length))
        for gap in [0.2]
        for coupling_length in [2.5, 4.5, 6.5]
    ]

    e += [
        uc.dbr_cavity_te(w0=w0, dw=dw)
        for w0 in [0.5]
        for dw in [50e-3, 100e-3, 150e-3, 200e-3]
    ]
    e += [add_gc(uc.ring_with_crossing())]
    e += [add_gc(uc.ring_with_crossing(port_name="o2", with_component=False))]

    c = gf.Component()
    _ = c << gf.pack(e, max_size=size, spacing=2)[0]
    _ = c << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return c


def test_ubc1() -> None:
    c = EBeam_JoaquinMatres_1()
    assert c


if __name__ == "__main__":
    c = EBeam_JoaquinMatres_1()
    c.show()  # show in klayout
