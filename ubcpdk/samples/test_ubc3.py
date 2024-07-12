import gdsfactory as gf

import ubcpdk.components as uc
from ubcpdk.tech import LAYER

size = (440, 470)
add_gc = uc.add_fiber_array


@gf.cell
def EBeam_JoaquinMatres_3() -> gf.Component:
    """Contains mirror cavities and structures inside a resonator."""
    e = []
    e += [add_gc(uc.ebeam_crossing4())]
    e += [add_gc(uc.ebeam_adiabatic_te1550(), optical_routing_type=1)]
    e += [add_gc(uc.ebeam_bdc_te1550())]
    e += [add_gc(uc.ebeam_y_1550(), optical_routing_type=1)]
    e += [add_gc(uc.straight(), component_name=f"straight_{i}") for i in range(2)]
    c = gf.Component()
    _ = c << gf.pack(e, max_size=size, spacing=2)[0]
    _ = c << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return c


def test_ubc3() -> None:
    c = EBeam_JoaquinMatres_3()
    assert c


if __name__ == "__main__":
    c = EBeam_JoaquinMatres_3()
    c.show()  # show in klayout
