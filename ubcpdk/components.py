"""Cells imported from the PDK."""

from functools import cache, partial

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.typings import (
    ComponentReference,
    ComponentSpec,
    CrossSectionSpec,
    Optional,
    Port,
)

from ubcpdk import tech
from ubcpdk.config import CONFIG, PATH
from ubcpdk.import_gds import import_gds
from ubcpdk.tech import (
    LAYER,
    add_pins_bbox_siepic,
)

um = 1e-6


@gf.cell
def bend_euler(cross_section="strip", **kwargs) -> Component:
    return gf.components.bend_euler(cross_section=cross_section, **kwargs)


bend_euler180 = partial(bend_euler, angle=180)
bend = bend_euler


def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        kwargs: additional cross_section arguments.

    .. code::

        o1 -------------- o2
                length
    """
    return gf.c.straight(
        length=length, npoints=npoints, cross_section=cross_section, **kwargs
    )


@gf.cell
def wire_corner(cross_section="metal_routing", **kwargs) -> Component:
    return gf.c.wire_corner(cross_section=cross_section, **kwargs)


@gf.cell
def straight_heater_metal(length: float = 320.0, cross_section="strip") -> gf.Component:
    c = gf.c.straight_heater_metal(length=length, cross_section=cross_section)
    return c


info1550te = dict(polarization="te", wavelength=1.55)
info1310te = dict(polarization="te", wavelength=1.31)
info1550tm = dict(polarization="tm", wavelength=1.55)
info1310tm = dict(polarization="tm", wavelength=1.31)
thermal_phase_shifter_names = [
    "thermal_phase_shifter_multimode_500um",
    "thermal_phase_shifter_te_1310_500um",
    "thermal_phase_shifter_te_1310_500um_lowloss",
    "thermal_phase_shifter_te_1550_500um_lowloss",
]

prefix_te1550 = prefix_tm1550 = prefix_te1310 = prefix_tm1130 = "o2"


def clean_name(name: str) -> str:
    return name.replace("_", ".")


def thermal_phase_shifter0() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        PATH.gds / "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[0]
    )


def thermal_phase_shifter1() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        PATH.gds / "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[1]
    )


def thermal_phase_shifter2() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        PATH.gds / "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[2]
    )


def thermal_phase_shifter3() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        PATH.gds / "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[3]
    )


def ebeam_BondPad() -> gf.Component:
    """Return ebeam_BondPad fixed cell."""
    return import_gds(PATH.gds / "ebeam_BondPad.gds")


def ebeam_adiabatic_te1550() -> gf.Component:
    """Return ebeam_adiabatic_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_adiabatic_te1550.gds")


def ebeam_adiabatic_tm1550() -> gf.Component:
    """Return ebeam_adiabatic_tm1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_adiabatic_tm1550.gds")


def ebeam_bdc_te1550() -> gf.Component:
    """Return ebeam_bdc_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_bdc_te1550.gds")


def ebeam_bdc_tm1550() -> gf.Component:
    """Return ebeam_bdc_tm1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_bdc_tm1550.gds")


def ebeam_crossing4() -> gf.Component:
    """Return ebeam_crossing4 fixed cell."""
    return import_gds(PATH.gds / "ebeam_crossing4.gds")


@gf.cell
def straight_one_pin(length=1, cross_section=tech.strip_bbox) -> gf.Component:
    c = gf.Component()
    add_pins_left = partial(tech.add_pins_siepic, prefix="o1", pin_length=0.1)
    s = c << gf.components.straight(length=length, cross_section=cross_section)
    c.add_ports(s.ports)
    add_pins_left(c)
    c.absorb(s)
    return c


@gf.cell
def ebeam_crossing4_2ports() -> gf.Component:
    """Return ebeam_crossing4 fixed cell."""
    c = gf.Component()
    x = c << ebeam_crossing4()
    s1 = c << straight_one_pin()
    s2 = c << straight_one_pin()

    s1.connect("o1", x.ports["o2"])
    s2.connect("o1", x.ports["o4"])

    c.add_port(name="o1", port=x.ports["o1"])
    c.add_port(name="o4", port=x.ports["o3"])
    c.flatten()
    return c


@gf.cell
def ebeam_splitter_adiabatic_swg_te1550() -> gf.Component:
    """Return ebeam_splitter_adiabatic_swg_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_splitter_adiabatic_swg_te1550.gds")


@gf.cell
def ebeam_splitter_swg_assist_te1310() -> gf.Component:
    """Return ebeam_splitter_swg_assist_te1310 fixed cell."""
    return import_gds(PATH.gds / "ebeam_splitter_swg_assist_te1310.gds")


@gf.cell
def ebeam_splitter_swg_assist_te1550() -> gf.Component:
    """Return ebeam_splitter_swg_assist_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_splitter_swg_assist_te1550.gds")


@gf.cell
def ebeam_swg_edgecoupler() -> gf.Component:
    """Return ebeam_swg_edgecoupler fixed cell."""
    return import_gds(PATH.gds / "ebeam_swg_edgecoupler.gds")


@gf.cell
def ebeam_terminator_te1310() -> gf.Component:
    """Return ebeam_terminator_te1310 fixed cell."""
    return import_gds(PATH.gds / "ebeam_terminator_te1310.gds")


@gf.cell
def ebeam_terminator_te1550() -> gf.Component:
    """Return ebeam_terminator_te1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_terminator_te1550.gds")


@gf.cell
def ebeam_terminator_tm1550() -> gf.Component:
    """Return ebeam_terminator_tm1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_terminator_tm1550.gds")


def ebeam_y_1550() -> gf.Component:
    """Return ebeam_y_1550 fixed cell."""
    return import_gds(PATH.gds / "ebeam_y_1550.gds")


@gf.cell
def ebeam_y_adiabatic() -> gf.Component:
    """Return ebeam_y_adiabatic fixed cell."""
    return import_gds(PATH.gds / "ebeam_y_adiabatic.gds")


@gf.cell
def ebeam_y_adiabatic_1310() -> gf.Component:
    """Return ebeam_y_adiabatic_1310 fixed cell."""
    return import_gds(PATH.gds / "ebeam_y_adiabatic_1310.gds")


@gf.cell
def metal_via() -> gf.Component:
    """Return metal_via fixed cell."""
    return import_gds(PATH.gds / "metal_via.gds")


@gf.cell
def photonic_wirebond_surfacetaper_1310() -> gf.Component:
    """Return photonic_wirebond_surfacetaper_1310 fixed cell."""
    return import_gds(PATH.gds / "photonic_wirebond_surfacetaper_1310.gds")


@gf.cell
def photonic_wirebond_surfacetaper_1550() -> gf.Component:
    """Return photonic_wirebond_surfacetaper_1550 fixed cell."""
    return import_gds(PATH.gds / "photonic_wirebond_surfacetaper_1550.gds")


@gf.cell
def gc_te1310() -> gf.Component:
    """Return ebeam_gc_te1310 fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1310.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1310te)
    c.flatten()
    return c


@gf.cell
def gc_te1310_8deg() -> gf.Component:
    """Return ebeam_gc_te1310_8deg fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1310_8deg.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1310te)
    c.flatten()
    return c


@gf.cell
def gc_te1310_broadband() -> gf.Component:
    """Return ebeam_gc_te1310_broadband fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1310_broadband.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1310te)
    c.flatten()
    return c


@gf.cell
def gc_te1550() -> gf.Component:
    """Return ebeam_gc_te1550 fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1550.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1550te)
    c.flatten()
    return c


@gf.cell
def gc_te1550_90nmSlab() -> gf.Component:
    """Return ebeam_gc_te1550_90nmSlab fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1550_90nmSlab.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1550te)
    c.flatten()
    return c


@gf.cell
def gc_te1550_broadband() -> gf.Component:
    """Return ebeam_gc_te1550_broadband fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_te1550_broadband.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type="vertical_te",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1550te)
    c.flatten()
    return c


@gf.cell
def gc_tm1550() -> gf.Component:
    """Return ebeam_gc_tm1550 fixed cell."""
    c = gf.Component()
    gc = import_gds(PATH.gds / "ebeam_gc_tm1550.gds")
    gc_ref = c << gc
    gc_ref.dmirror()
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_tm1550
    c.add_port(
        name=name,
        port_type="vertical_tm",
        center=(25, 0),
        layer=(1, 0),
        width=9,
        orientation=0,
    )
    c.info.update(info1550tm)
    c.flatten()
    return c


mzi = partial(
    gf.components.mzi,
    splitter=ebeam_y_1550,
    bend=bend_euler,
    straight="straight",
    cross_section="strip",
)

_mzi_heater = partial(
    gf.components.mzi_phase_shifter,
    bend="bend_euler",
    straight="straight",
    splitter="ebeam_y_1550",
    straight_x_top="straight_heater_metal",
)


@gf.cell
def mzi_heater(delta_length=10.0, length_x=320) -> gf.Component:
    """Returns MZI with heater.

    Args:
        delta_length: extra length for mzi arms.
    """
    c = _mzi_heater(delta_length=delta_length, length_x=length_x)
    return c


@gf.cell
def via_stack_heater_mtop(size=(10, 10)) -> gf.Component:
    return gf.components.via_stack(
        size=size,
        layers=(LAYER.M1_HEATER, LAYER.M2_ROUTER),
        vias=(None, None),
    )


def get_input_label_text(
    port: Port,
    gc: ComponentReference,
    component_name: Optional[str] = None,
    username: str = CONFIG.username,
) -> str:
    """Return label for port and a grating coupler.

    Args:
        port: component port.
        gc: grating coupler reference.
        component_name: optional component name.
        username: for the label.
    """
    polarization = gc.info.get("polarization")
    wavelength = gc.info.get("wavelength")

    assert polarization.upper() in [
        "TE",
        "TM",
    ], f"Not valid polarization {polarization.upper()!r} in [TE, TM]"
    assert (
        isinstance(wavelength, int | float) and 1.0 < wavelength < 2.0
    ), f"{wavelength} is Not valid 1000 < wavelength < 2000"

    name = component_name
    name = clean_name(name)
    # return f"opt_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}-{name}-{gc_index}-{port.name}"
    return f"opt_in_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}-{name}"


def add_fiber_array(
    component: ComponentSpec = straight,
    component_name: Optional[str] = None,
    gc_port_name: str = "o1",
    with_loopback: bool = False,
    optical_routing_type: int = 1,
    fanout_length: float = 0.0,
    grating_coupler: ComponentSpec = gc_te1550,
    cross_section: CrossSectionSpec = "strip",
    straight: ComponentSpec = "straight",
    taper: ComponentSpec | None = None,
    **kwargs,
) -> Component:
    """Returns component with grating couplers and labels on each port.

    Routes all component ports south.
    Can add align_ports loopback reference structure on the edges.

    Args:
        component: to connect.
        component_name: for the label.
        gc_port_name: grating coupler input port name 'o1'.
        with_loopback: True, adds loopback structures.
        optical_routing_type: None: autoselection, 0: no extension.
        fanout_length: None  # if None, automatic calculation of fanout length.
        grating_coupler: grating coupler instance, function or list of functions.
        cross_section: spec.
        straight: straight component.
        taper: taper component.
        kwargs: cross_section settings.

    """
    c = gf.Component()

    ref = c << gf.routing.add_fiber_array(
        straight=straight,
        bend=bend,
        component=component,
        component_name=component_name,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        with_loopback=with_loopback,
        optical_routing_type=optical_routing_type,
        fanout_length=fanout_length,
        cross_section=cross_section,
        taper=taper,
        **kwargs,
    )
    ref.drotate(-90)
    c.add_ports(ref.ports)
    c.copy_child_info(component)
    return c


L = 1.55 / 4 / 2 / 2.44


@gf.cell
def dbg(
    w0: float = 0.5,
    dw: float = 0.1,
    n: int = 100,
    l1: float = L,
    l2: float = L,
) -> gf.Component:
    """Includes two ports.

    Args:
        w0: width.
        dw: delta width.
        n: number of elements.
        l1: length teeth1.
        l2: length teeth2.
    """
    c = gf.Component()
    s = gf.components.straight(length=l1, cross_section="strip")
    g = c << gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        cross_section="strip",
    )
    s1 = c << s
    s2 = c << s
    s1.connect("o2", g.ports["o1"])
    s2.connect("o2", g.ports["o2"])

    c.add_port("o1", port=s1.ports["o1"])
    c.add_port("o2", port=s2.ports["o1"])
    c = add_pins_bbox_siepic(c)
    return c


@gf.cell
def terminator_short(width2=0.1) -> gf.Component:
    c = gf.Component()
    s = gf.components.taper(cross_section="strip", width2=width2)
    s1 = c << s
    c.add_port("o1", port=s1.ports["o1"])
    c = add_pins_bbox_siepic(c)
    c.flatten()
    return c


@gf.cell
def dbr(
    w0: float = 0.5,
    dw: float = 0.1,
    n: int = 100,
    l1: float = L,
    l2: float = L,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> gf.Component:
    """Returns distributed bragg reflector.

    Args:
        w0: width.
        dw: delta width.
        n: number of elements.
        l1: length teeth1.
        l2: length teeth2.
        cross_section: spec.
        kwargs: cross_section settings.
    """
    c = gf.Component()

    xs = gf.get_cross_section(cross_section, **kwargs)

    # add_pins_left = partial(add_pins_siepic, prefix="o1")
    s = c << gf.components.straight(length=l1, cross_section=xs)
    _dbr = gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        cross_section=xs,
    )
    dbr = c << _dbr
    s.connect("o2", dbr.ports["o1"])
    c.add_port("o1", port=s.ports["o1"])
    return add_pins_bbox_siepic(c)


@gf.cell
def coupler(**kwargs) -> gf.Component:
    c = gf.components.coupler(**kwargs).dup()
    c.flatten()
    return c


@gf.cell
def mmi1x2(**kwargs) -> gf.Component:
    return gf.components.mmi1x2(**kwargs)


@gf.cell
def dbr_cavity(dbr=dbr, coupler="coupler", **kwargs) -> gf.Component:
    dbr = dbr(**kwargs)
    return gf.components.cavity(component=dbr, coupler=coupler)


@cache
def dbr_cavity_te(component="dbr_cavity", **kwargs) -> gf.Component:
    component = gf.get_component(component, **kwargs)
    return add_fiber_array(component=component)


@gf.cell
def spiral(
    length: float = 100,
    spacing: float = 3.0,
    n_loops: int = 6,
) -> gf.Component:
    """Returns spiral component.

    Args:
        length: length.
        spacing: spacing.
        n_loops: number of loops.
    """
    return gf.c.spiral(
        length=length,
        spacing=spacing,
        n_loops=n_loops,
        bend=bend_euler,
        straight=straight,
    )


coupler90 = partial(gf.components.coupler90, bend=bend_euler, straight=straight)
coupler_straight = partial(
    gf.components.coupler_straight, gap=0.2, cross_section="strip"
)


@gf.cell
def coupler_ring(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_extension: float = 3,
    bend=bend,
    cross_section="strip",
    **kwargs,
) -> Component:
    c = gf.components.coupler_ring(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_extension=length_extension,
        bend=bend,
        cross_section=cross_section,
        **kwargs,
    ).dup()
    c = add_pins_bbox_siepic(c)
    c.flatten()
    return c


@gf.cell
def ring_single(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
) -> Component:
    return gf.components.ring_single(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        cross_section="strip",
        bend=bend,
        coupler_ring=coupler_ring,
    )


@gf.cell
def ring_double(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
) -> Component:
    return gf.components.ring_double(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        cross_section="strip",
        bend=bend,
        coupler_ring=coupler_ring,
    )


ring_double_heater = partial(
    gf.components.ring_double_heater,
    via_stack="via_stack_heater_mtop",
    straight=straight,
    length_y=0.2,
    cross_section_heater="heater_metal",
    cross_section_waveguide_heater="strip_heater_metal",
    cross_section="strip",
    coupler_ring=coupler_ring,
)
ring_single_heater = partial(
    gf.components.ring_single_heater,
    via_stack="via_stack_heater_mtop",
    straight=straight,
    cross_section_heater="heater_metal",
    cross_section_waveguide_heater="strip_heater_metal",
    cross_section="strip",
    coupler_ring=coupler_ring,
)


ebeam_dc_te1550 = partial(
    gf.components.coupler,
)
taper = partial(gf.components.taper)
ring_with_crossing = partial(
    gf.components.ring_single_dut,
    component=ebeam_crossing4_2ports,
    port_name="o4",
    bend=bend,
    cross_section="strip",
)


pad = partial(
    gf.components.pad,
    size=(75, 75),
    layer=LAYER.M2_ROUTER,
    bbox_layers=(LAYER.PAD_OPEN,),
    bbox_offsets=(-1.8,),
)


def add_label_electrical(component: Component, text: str, port_name: str = "e2"):
    """Adds labels for electrical port.

    Returns same component so it needs to be used as a decorator.
    """
    if port_name not in component.ports:
        port_names = [port.name for port in component.ports]
        raise ValueError(f"No port {port_name!r} in {port_names}")

    component.add_label(
        text=text, position=component.ports[port_name].center, layer=LAYER.TEXT
    )
    return component


pad_array = partial(gf.components.pad_array, pad=pad, spacing=(125, 125))
add_pads_rf = partial(
    gf.routing.add_electrical_pads_top,
    component="ring_single_heater",
    pad_array="pad_array",
)
add_pads_top = partial(
    gf.routing.add_pads_top,
    component=straight_heater_metal,
)
add_pads_bot = partial(
    gf.routing.add_pads_bot,
    component=straight_heater_metal,
)


@cache
def add_fiber_array_pads_rf(
    component: ComponentSpec = "ring_single_heater",
    username: str = CONFIG.username,
    orientation: float = 0,
    **kwargs,
) -> Component:
    """Returns fiber array with label and electrical pads.

    Args:
        component: to add fiber array and pads.
        username: for the label.
        orientation: for adding pads.
        kwargs: for add_fiber_array.
    """
    c0 = gf.get_component(component)
    # text = f"elec_{username}-{clean_name(c0.name)}_G"
    # add_label = partial(add_label_electrical, text=text)
    c1 = add_pads_rf(component=c0, orientation=orientation)

    # ports_names = [port.name for port in c0.ports if port.orientation == orientation]
    # c1 = add_pads_top(component=c0, port_names=ports_names)
    return add_fiber_array(component=c1, **kwargs)


@cache
def add_pads(
    component: ComponentSpec = "ring_single_heater",
    username: str = CONFIG.username,
    **kwargs,
) -> Component:
    """Returns fiber array with label and electrical pads.

    Args:
        component: to add fiber array and pads.
        username: for the label.
        kwargs: for add_fiber_array.
    """
    c0 = gf.get_component(component)
    text = f"elec_{username}-{clean_name(c0.name)}_G"
    c0._locked = False
    c0 = add_label_electrical(c0, text=text)
    return add_pads_rf(component=c0, **kwargs)


if __name__ == "__main__":
    # c = straight_heater_metal()
    # c = thermal_phase_shifter0()
    # c = straight_one_pin()
    # c = ebeam_adiabatic_te1550()
    # c = ebeam_bdc_te1550()
    # c = gc_tm1550()
    # c = spiral()
    # c = add_pads_top()

    # c.pprint_ports()
    # c.pprint_ports()
    # c = straight()
    # c = terminator_short()
    # c = add_fiber_array_pads_rf(c)

    # c = ring_double(length_y=10)
    # c = ring_with_crossing()
    # c = straight_heater_metal()
    # c = add_fiber_array(straight_heater_metal)
    # c.pprint_ports()
    # c = coupler_ring()
    # c = dbr_cavity_te()
    # c = dbr_cavity()
    # c = ring_single(radius=12)
    # c = bend_euler()
    # c = mzi()
    # c = spiral()
    # c = pad_array()
    # c = bend_euler()
    c = ebeam_y_1550()
    c = ebeam_y_1550()
    # c = mzi_heater()
    # c = ring_with_crossing()
    # c = ring_single()
    # c = ring_double()
    # c = ring_double(radius=12, length_x=2, length_y=2)
    # c = straight()
    c.show()
