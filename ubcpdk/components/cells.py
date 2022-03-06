"""

Each partial function is equivalent to

def y_splitter() -> Component:
    c = import_gds("ebeam_y_1550", rename_ports=True)
    return c
"""

import gdsfactory as gf
from ubcpdk.import_gds import import_gds
from ubcpdk.add_pins import add_pins

from ubcpdk.components.straight import straight


dc_broadband_te = gf.partial(
    import_gds,
    "ebeam_bdc_te1550.gds",
    doc="Broadband directional coupler TE1550 50/50 power.",
)

dc_broadband_tm = gf.partial(
    import_gds,
    "ebeam_bdc_tm1550.gds",
    doc="Broadband directional coupler TM1550 50/50 power.",
)

dc_adiabatic = gf.partial(
    import_gds,
    "ebeam_adiabatic_te1550.gds",
    doc="Adiabatic directional coupler TE1550 50/50 power.",
)

y_adiabatic = gf.partial(
    import_gds,
    "ebeam_y_adiabatic.gds",
    doc="Adiabatic Y junction TE1550 50/50 power.",
)

y_splitter = gf.partial(
    import_gds,
    "ebeam_y_1550.gds",
    doc="Y junction TE1550 50/50 power.",
)

bend_euler = gf.partial(gf.components.bend_euler, decorator=add_pins)
mzi = gf.partial(
    gf.components.mzi, splitter=y_splitter, straight=straight, bend=bend_euler
)
ring_single = gf.partial(gf.components.ring_single)
ebeam_dc_halfring_straight = gf.partial(gf.components.coupler_ring)
ebeam_dc_te1550 = gf.partial(gf.components.coupler)
spiral = gf.partial(gf.components.spiral_external_io)


if __name__ == "__main__":
    # print(dc_broadband_te.__doc__)
    # c = dc_broadband_te()
    # c = dc_adiabatic()
    # c = straight_no_pins()
    # c = add_fiber_array(component=c)
    # c = gc_tm1550()
    # print(c.get_ports_array())
    # print(c.ports.keys())
    # c = straight()
    # c = add_fiber_array(component=c)
    c = mzi()
    # c = gc_te1550()
    c.show()
