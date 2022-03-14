""" Each partial function is equivalent to

def y_splitter() -> Component:
    c = import_gds("ebeam_y_1550", rename_ports=True)
    return c
"""

import gdsfactory as gf
from gdsfactory.add_pins import add_pins_bbox_siepic

from ubcpdk.import_gds import import_gds_siepic_pins
from ubcpdk.components.straight import straight
from ubcpdk.tech import strip, LAYER_STACK, LAYER


dc_broadband_te = gf.partial(
    import_gds_siepic_pins,
    "ebeam_bdc_te1550.gds",
    doc="Broadband directional coupler TE1550 50/50 power.",
)

dc_broadband_tm = gf.partial(
    import_gds_siepic_pins,
    "ebeam_bdc_tm1550.gds",
    doc="Broadband directional coupler TM1550 50/50 power.",
)

dc_adiabatic = gf.partial(
    import_gds_siepic_pins,
    "ebeam_adiabatic_te1550.gds",
    doc="Adiabatic directional coupler TE1550 50/50 power.",
)

y_adiabatic = gf.partial(
    import_gds_siepic_pins,
    "ebeam_y_adiabatic.gds",
    doc="Adiabatic Y junction TE1550 50/50 power.",
    name="ebeam_y_adiabatic",
)

y_splitter = gf.partial(
    import_gds_siepic_pins,
    "ebeam_y_1550.gds",
    doc="Y junction TE1550 50/50 power.",
    name="ebeam_y_1550",
    model="ebeam_y_1550",
    opt1="opt_a1",
    opt2="opt_b1",
    opt3="opt_b2",
)
crossing = gf.partial(
    import_gds_siepic_pins,
    "ebeam_crossing4.gds",
    doc="TE waveguide crossing.",
)


bend_euler = gf.partial(gf.components.bend_euler, decorator=add_pins_bbox_siepic)
mzi = gf.partial(
    gf.components.mzi,
    splitter=y_splitter,
    straight=straight,
    bend=bend_euler,
    port_e1_splitter="opt2",
    port_e0_splitter="opt3",
    port_e1_combiner="opt2",
    port_e0_combiner="opt3",
)
ring_single = gf.partial(gf.components.ring_single)

@gf.cell
def ebeam_dc_halfring_straight(
        gap: float = 0.2,
        radius: float = 5.0,
        length_x: float = 4.0,
        cross_section=strip,
        **kwargs
):
    component = gf.components.coupler_ring(
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
    )
    x = cross_section(**kwargs) if callable(cross_section) else cross_section
    thickness = LAYER_STACK.get_layer_to_thickness()
    um = 1e-6
    component.info["model"] = "ebeam_dc_halfring_straight"
    component.info["name"] = "ebeam_dc_halfring_straight"
    component.info["o1"] = "port 1"
    component.info["o2"] = "port 2"
    component.info["o3"] = "port 4"
    component.info["o4"] = "port 3"
    component.info["interconnect"] = {
        "gap": gap*um,
        "radius": radius*um,
        "wg_thickness": thickness[LAYER.WG]*um,
        "wg_width": x.info["width"]*um,
        "Lc": length_x*um
    }
    return component


ebeam_dc_te1550 = gf.partial(gf.components.coupler)
spiral = gf.partial(gf.components.spiral_external_io)
ring_with_crossing = gf.partial(
    gf.components.ring_single_dut, component=crossing, port_name="opt4"
)


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
    # c = mzi(splitter=y_splitter)
    # c = gc_te1550()

    # c = y_splitter()
    # s = dc_adiabatic()

    # c = mzi()

    # c = gf.Component()
    # s = y_splitter()
    # sp = c << s
    # wg = c << straight()
    # wg.connect("o1", sp.ports["opt1"])

    c = ebeam_dc_halfring_straight()
    c.show(show_ports=False)
