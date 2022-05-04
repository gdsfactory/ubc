""" Each partial function is equivalent to

def y_splitter() -> Component:
    c = import_gds("ebeam_y_1550", rename_ports=True)
    return c
"""

import gdsfactory as gf
from gdsfactory.add_labels import add_siepic_labels
from gdsfactory.add_pins import add_pins_bbox_siepic

from ubcpdk.import_gds import import_gds
from ubcpdk.tech import strip, LAYER_STACK, LAYER
from ubcpdk.import_gds import remove_pins_recursive


um = 1e-6

add_siepic_labels = gf.partial(
    add_siepic_labels, label_layer=LAYER.DEVREC, library="Design Kits/ebeam"
)

straight_ubc = gf.partial(gf.components.straight, cross_section="strip")
bend_euler_ubc = gf.partial(gf.components.bend_euler, cross_section="strip")
bend_s_ubc = gf.partial(gf.components.bend_s, cross_section="strip")

straight = gf.compose(add_siepic_labels, straight_ubc)
bend_euler = gf.compose(add_siepic_labels, bend_euler_ubc)
bend_s = gf.compose(add_siepic_labels, bend_s_ubc)


dc_broadband_te = gf.partial(
    import_gds,
    "ebeam_bdc_te1550.gds",
    name="ebeam_bdc_te1550",
    model="ebeam_bdc_te1550",
    doc="Broadband directional coupler TE1550 50/50 power.",
)

dc_broadband_tm = gf.partial(
    import_gds,
    "ebeam_bdc_tm1550.gds",
    name="ebeam_bdc_tm1550",
    model="ebeam_bdc_tm1550",
    doc="Broadband directional coupler TM1550 50/50 power.",
)

dc_adiabatic = gf.partial(
    import_gds,
    "ebeam_adiabatic_te1550.gds",
    name="ebeam_adiabatic_te1550",
    model="ebeam_adiabatic_te1550",
    doc="Adiabatic directional coupler TE1550 50/50 power.",
)

y_adiabatic = gf.partial(
    import_gds,
    "ebeam_y_adiabatic.gds",
    name="ebeam_y_adiabatic",
    model="ebeam_y_adiabatic",
    doc="Adiabatic Y junction TE1550 50/50 power.",
)

y_splitter = gf.partial(
    import_gds,
    "ebeam_y_1550.gds",
    doc="Y junction TE1550 50/50 power.",
    name="ebeam_y_1550",
    model="ebeam_y_1550",
    layout_model_port_pairs=(
        ("opt1", "opt_a1"),
        ("opt2", "opt_b1"),
        ("opt3", "opt_b2"),
    ),
)
crossing = gf.partial(
    import_gds,
    "ebeam_crossing4.gds",
    model="ebeam_crossing4",
    doc="TE waveguide crossing.",
)

mzi = gf.partial(
    gf.components.mzi,
    splitter=y_splitter,
    bend=bend_euler,
    straight=straight,
    port_e1_splitter="opt2",
    port_e0_splitter="opt3",
    port_e1_combiner="opt2",
    port_e0_combiner="opt3",
    cross_section="strip",
)


@gf.cell
def ebeam_dc_halfring_straight(
    gap: float = 0.2,
    radius: float = 5.0,
    length_x: float = 4.0,
    cross_section="strip",
    siepic: bool = True,
    model: str = "ebeam_dc_halfring_straight",
    **kwargs
):
    c = gf.Component()
    coupler_ring = c << gf.components.coupler_ring(
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend_euler,
        straight=straight,
        cross_section=cross_section,
    )
    x = gf.get_cross_section(cross_section=cross_section, **kwargs)
    thickness = LAYER_STACK.get_layer_to_thickness()

    c.add_ports(coupler_ring.ports)

    if siepic:
        c.info.update(
            layout_model_port_pairs=(
                ("o1", "port 1"),
                ("o2", "port 2"),
                ("o3", "port 4"),
                ("o4", "port 3"),
            ),
            properties={
                "gap": gap * um,
                "radius": radius * um,
                "wg_thickness": thickness[LAYER.WG] * um,
                "wg_width": x.width * um,
                "Lc": length_x * um,
            },
            component_type=["optical"],
        )

        add_siepic_info = gf.compose(
            gf.partial(add_siepic_labels, model=model),
            add_pins_bbox_siepic,
            remove_pins_recursive,
        )
        c = add_siepic_info(c)
    return c


ebeam_dc_te1550 = gf.compose(
    gf.partial(add_siepic_labels, model="ebeam_dc_te1550"),
    add_pins_bbox_siepic,
    remove_pins_recursive,
    gf.partial(
        gf.components.coupler,
        # component_type=["optical"],
        # layout_model_property_pairs=(("length", "coupling_length")),
        # properites=(("annotate", False)),
    ),
)
spiral = gf.partial(gf.components.spiral_external_io)
ring_with_crossing = gf.partial(
    gf.components.ring_single_dut,
    component=crossing,
    port_name="opt4",
    bend=bend_euler,
    straight=straight,
    cross_section=strip,
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

    # c = gf.Component()
    # s = y_splitter()
    # sp = c << s
    # wg = c << straight()
    # wg.connect("o1", sp.ports["opt1"])

    # c = ebeam_dc_halfring_straight()
    c = ring_with_crossing()

    c = mzi()
    c = ebeam_dc_te1550()
    c.show()
