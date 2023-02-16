"""Cells imported from the PDK."""
from functools import partial
import gdsfactory as gf
from gdsfactory.typings import (
    Callable,
    ComponentReference,
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
    List,
    Optional,
    Port,
    Tuple,
    Label,
)
from gdsfactory import Component

from ubcpdk.config import CONFIG
from ubcpdk.import_gds import import_gds, import_gc
from ubcpdk import tech
from ubcpdk.tech import (
    strip,
    LAYER_STACK,
    LAYER,
    add_pins_bbox_siepic,
    add_pins_bbox_siepic_remove_layers,
    add_pins_siepic_metal,
)

um = 1e-6


straight = gf.partial(
    gf.components.straight,
    cross_section="strip",
)
bend_euler = gf.partial(gf.components.bend_euler, cross_section="strip", npoints=100)
bend_s = gf.partial(
    gf.components.bend_s,
    cross_section="strip",
)

info1550te = dict(polarization="te", wavelength=1.55)
info1310te = dict(polarization="te", wavelength=1.31)
info1550tm = dict(polarization="tm", wavelength=1.55)
info1310tm = dict(polarization="tm", wavelength=1.31)
thermal_phase_shifter_names = [
    "thermal_phase_shifter_m_6480beac",
    "thermal_phase_shifter_t_22d678c3",
    "thermal_phase_shifter_t_75acd1c1",
    "thermal_phase_shifter_t_ab7ae757",
]

prefix_te1550 = f"opt_in_TE_1550_device_{CONFIG.username}"
prefix_tm1550 = f"opt_in_TM_1550_device_{CONFIG.username}"
prefix_te1310 = f"opt_in_TE_1310_device_{CONFIG.username}"
prefix_tm1130 = f"opt_in_TM_1310_device_{CONFIG.username}"


# @gf.cell
# def Packaging_FibreArray_8ch() -> gf.Component:
#     """Return Packaging_FibreArray_8ch fixed cell."""
#     return import_gds("Packaging_FibreArray_8ch.gds")


def clean_name(name: str) -> str:
    return name.replace("_", ".")


@gf.cell
def thermal_phase_shifter0() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[0]
    )


@gf.cell
def thermal_phase_shifter1() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[1]
    )


@gf.cell
def thermal_phase_shifter2() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[2]
    )


@gf.cell
def thermal_phase_shifter3() -> gf.Component:
    """Return thermal_phase_shifters fixed cell."""
    return import_gds(
        "thermal_phase_shifters.gds", cellname=thermal_phase_shifter_names[3]
    )


@gf.cell
def ebeam_BondPad() -> gf.Component:
    """Return ebeam_BondPad fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_BondPad()
      c.plot()
    """
    return import_gds("ebeam_BondPad.gds")


@gf.cell
def ebeam_adiabatic_te1550() -> gf.Component:
    """Return ebeam_adiabatic_te1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_adiabatic_te1550()
      c.plot()
    """
    return import_gds("ebeam_adiabatic_te1550.gds")


@gf.cell
def ebeam_adiabatic_tm1550() -> gf.Component:
    """Return ebeam_adiabatic_tm1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_adiabatic_tm1550()
      c.plot()
    """
    return import_gds("ebeam_adiabatic_tm1550.gds")


@gf.cell
def ebeam_bdc_te1550() -> gf.Component:
    """Return ebeam_bdc_te1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_bdc_te1550()
      c.plot()
    """
    return import_gds("ebeam_bdc_te1550.gds")


@gf.cell
def ebeam_bdc_tm1550() -> gf.Component:
    """Return ebeam_bdc_tm1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_bdc_tm1550()
      c.plot()
    """
    return import_gds("ebeam_bdc_tm1550.gds")


@gf.cell
def ebeam_crossing4() -> gf.Component:
    """Return ebeam_crossing4 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_crossing4()
      c.plot()
    """
    return import_gds("ebeam_crossing4.gds")


@gf.cell
def straight_one_pin(length=1, cross_section=tech.strip_bbox_only) -> gf.Component:
    c = gf.Component()
    add_pins_left = partial(tech.add_pins_siepic, prefix="o1", pin_length=0.1)
    s = c << gf.components.straight(length=length, cross_section=cross_section)
    c.add_ports(s.ports)
    add_pins_left(c)
    c.absorb(s)
    return c


@gf.cell
def ebeam_crossing4_2ports() -> gf.Component:
    """Return ebeam_crossing4 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_crossing4_2ports()
      c.plot()
    """
    c = gf.Component()
    x = c << ebeam_crossing4()
    s1 = c << straight_one_pin()
    s2 = c << straight_one_pin()

    s1.connect("o1", x.ports["o2"])
    s2.connect("o1", x.ports["o4"])

    c.add_port(name="o1", port=x.ports["o1"])
    c.add_port(name="o4", port=x.ports["o3"])
    return c


@gf.cell
def ebeam_splitter_adiabatic_swg_te1550() -> gf.Component:
    """Return ebeam_splitter_adiabatic_swg_te1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_splitter_adiabatic_swg_te1550()
      c.plot()
    """
    return import_gds("ebeam_splitter_adiabatic_swg_te1550.gds")


@gf.cell
def ebeam_splitter_swg_assist_te1310() -> gf.Component:
    """Return ebeam_splitter_swg_assist_te1310 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_splitter_swg_assist_te1310()
      c.plot()
    """
    return import_gds("ebeam_splitter_swg_assist_te1310.gds")


@gf.cell
def ebeam_splitter_swg_assist_te1550() -> gf.Component:
    """Return ebeam_splitter_swg_assist_te1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_splitter_swg_assist_te1550()
      c.plot()
    """
    return import_gds("ebeam_splitter_swg_assist_te1550.gds")


@gf.cell
def ebeam_swg_edgecoupler() -> gf.Component:
    """Return ebeam_swg_edgecoupler fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_swg_edgecoupler()
      c.plot()
    """
    return import_gds("ebeam_swg_edgecoupler.gds")


@gf.cell
def ebeam_terminator_te1310() -> gf.Component:
    """Return ebeam_terminator_te1310 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_terminator_te1310()
      c.plot()
    """
    return import_gds("ebeam_terminator_te1310.gds")


@gf.cell
def ebeam_terminator_te1550() -> gf.Component:
    """Return ebeam_terminator_te1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_terminator_te1550()
      c.plot()
    """
    return import_gds("ebeam_terminator_te1550.gds")


@gf.cell
def ebeam_terminator_tm1550() -> gf.Component:
    """Return ebeam_terminator_tm1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_terminator_tm1550()
      c.plot()
    """
    return import_gds("ebeam_terminator_tm1550.gds")


@gf.cell
def ebeam_y_1550() -> gf.Component:
    """Return ebeam_y_1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_y_1550()
      c.plot()
    """
    return import_gds("ebeam_y_1550.gds")


@gf.cell
def ebeam_y_adiabatic() -> gf.Component:
    """Return ebeam_y_adiabatic fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_y_adiabatic()
      c.plot()
    """
    return import_gds("ebeam_y_adiabatic.gds")


@gf.cell
def ebeam_y_adiabatic_tapers() -> gf.Component:
    """Return ebeam_y_adiabatic fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_y_adiabatic()
      c.plot()
    """
    y = import_gds("ebeam_y_adiabatic.gds")
    return gf.add_tapers(y)


@gf.cell
def ebeam_y_adiabatic_1310() -> gf.Component:
    """Return ebeam_y_adiabatic_1310 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_y_adiabatic_1310()
      c.plot()
    """
    return import_gds("ebeam_y_adiabatic_1310.gds")


@gf.cell
def metal_via() -> gf.Component:
    """Return metal_via fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.metal_via()
      c.plot()
    """
    return import_gds("metal_via.gds")


@gf.cell
def photonic_wirebond_surfacetaper_1310() -> gf.Component:
    """Return photonic_wirebond_surfacetaper_1310 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.photonic_wirebond_surfacetaper_1310()
      c.plot()
    """
    return import_gds("photonic_wirebond_surfacetaper_1310.gds")


@gf.cell
def photonic_wirebond_surfacetaper_1550() -> gf.Component:
    """Return photonic_wirebond_surfacetaper_1550 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.photonic_wirebond_surfacetaper_1550()
      c.plot()
    """
    return import_gds("photonic_wirebond_surfacetaper_1550.gds")


@gf.cell
def gc_te1310() -> gf.Component:
    """Return ebeam_gc_te1310 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.gc_te1310()
      c.plot()
    """
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1310.gds", info=info1310te)
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type=name,
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    return c


@gf.cell
def gc_te1310_8deg() -> gf.Component:
    """Return ebeam_gc_te1310_8deg fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.gc_te1310_8deg()
      c.plot()
    """
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1310_8deg.gds", info=info1310te)
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type=name,
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    return c


@gf.cell
def gc_te1310_broadband() -> gf.Component:
    """Return ebeam_gc_te1310_broadband fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.gc_te1310_broadband()
      c.plot()
    """
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1310_broadband.gds", info=info1310te)
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1310
    c.add_port(
        name=name,
        port_type=name,
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    return c


@gf.cell
def gc_te1550() -> gf.Component:
    """Return ebeam_gc_te1550 fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1550.gds", info=info1550te)
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type=name,
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    return c


@gf.cell
def gc_te1550_90nmSlab() -> gf.Component:
    """Return ebeam_gc_te1550_90nmSlab fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1550_90nmSlab.gds", info=info1550te)
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type=name,
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    return c


@gf.cell
def gc_te1550_broadband() -> gf.Component:
    """Return ebeam_gc_te1550_broadband fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_te1550_broadband.gds", info=info1550te)
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_te1550
    c.add_port(
        name=name,
        port_type=name,
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    return c


@gf.cell
def gc_tm1550() -> gf.Component:
    """Return ebeam_gc_tm1550 fixed cell."""
    c = gf.Component()
    gc = import_gc("ebeam_gc_tm1550.gds", info=info1550tm)
    gc_ref = c << gc
    c.add_ports(gc_ref.ports)
    c.copy_child_info(gc)
    name = prefix_tm1550
    c.add_port(
        name=name,
        port_type=name,
        center=(25, 0),
        layer=(1, 0),
        width=9,
    )
    return c


mzi = gf.partial(
    gf.components.mzi,
    splitter=ebeam_y_1550,
    bend=bend_euler,
    straight=straight,
    cross_section="strip",
)

mzi_heater = gf.partial(
    gf.components.mzi_phase_shifter,
    splitter=ebeam_y_1550,
)

via_stack_heater_mtop = gf.partial(
    gf.components.via_stack,
    size=(10, 10),
    layers=(LAYER.M1_HEATER, LAYER.M2_ROUTER),
    vias=(None, None),
)
ring_double_heater = gf.partial(
    gf.components.ring_double_heater, via_stack=via_stack_heater_mtop
)
ring_single_heater = gf.partial(
    gf.components.ring_single_heater,
    via_stack=via_stack_heater_mtop,
)


def get_input_label_text(
    port: Port,
    gc: ComponentReference,
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
    username: str = CONFIG.username,
) -> str:
    """Return label for port and a grating coupler.

    Args:
        port: component port.
        gc: grating coupler reference.
        gc_index: grating coupler index.
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
        isinstance(wavelength, (int, float)) and 1.0 < wavelength < 2.0
    ), f"{wavelength} is Not valid 1000 < wavelength < 2000"

    name = component_name or port.parent.metadata_child.get("name")
    name = clean_name(name)
    # return f"opt_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}-{name}-{gc_index}-{port.name}"
    return f"opt_{polarization.upper()}_{int(wavelength * 1000.0)}_device_{username}-{name}"


def get_input_labels(
    io_gratings: List[ComponentReference],
    ordered_ports: List[Port],
    component_name: str,
    layer_label: Tuple[int, int] = (10, 0),
    gc_port_name: str = "o1",
    port_index: int = 1,
    get_input_label_text_function: Callable = get_input_label_text,
) -> List[Label]:
    """Return list of labels for all component ports.

    Args:
        io_gratings: list of grating_coupler references.
        ordered_ports: list of ports.
        component_name: name.
        layer_label: for the label.
        gc_port_name: grating_coupler port.
        port_index: index of the port.
        get_input_label_text_function: function.

    """
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_input_label_text(
        port=port, gc=gc, gc_index=port_index, component_name=component_name
    )
    layer, texttype = gf.get_layer(layer_label)
    label = Label(
        text=text,
        origin=gc.ports[gc_port_name].center,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


@gf.cell
def add_fiber_array(
    component: ComponentSpec = straight,
    component_name: Optional[str] = None,
    gc_port_name: str = "o1",
    get_input_labels_function: Callable = get_input_labels,
    with_loopback: bool = False,
    optical_routing_type: int = 0,
    fanout_length: float = 0.0,
    grating_coupler: ComponentSpec = gc_te1550,
    cross_section: CrossSectionSpec = "strip",
    layer_label: LayerSpec = (10, 0),
    **kwargs,
) -> Component:
    """Returns component with grating couplers and labels on each port.

    Routes all component ports south.
    Can add align_ports loopback reference structure on the edges.

    Args:
        component: to connect.
        component_name: for the label.
        gc_port_name: grating coupler input port name 'o1'.
        get_input_labels_function: function to get input labels for grating couplers.
        with_loopback: True, adds loopback structures.
        optical_routing_type: None: autoselection, 0: no extension.
        fanout_length: None  # if None, automatic calculation of fanout length.
        grating_coupler: grating coupler instance, function or list of functions.
        cross_section: spec.
        layer_label: for label.

    """
    c = gf.Component()

    component = gf.routing.add_fiber_array(
        component=component,
        component_name=component_name,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        get_input_labels_function=get_input_labels_function,
        get_input_label_text_function=get_input_label_text,
        with_loopback=with_loopback,
        optical_routing_type=optical_routing_type,
        layer_label=layer_label,
        fanout_length=fanout_length,
        cross_section=cross_section,
        **kwargs,
    )
    ref = c << component
    ref.rotate(-90)
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
    s = gf.components.straight(length=l1, cross_section=tech.strip_simple)
    g = c << gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        add_pins=None,
        cross_section=tech.strip_simple,
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
def terminator_short(**kwargs) -> gf.Component:
    c = gf.Component()
    s = gf.components.taper(**kwargs, cross_section=tech.strip_simple)
    s1 = c << s
    c.add_port("o1", port=s1.ports["o1"])
    c = add_pins_bbox_siepic(c)
    return c


@gf.cell
def dbr(
    w0: float = 0.5,
    dw: float = 0.1,
    n: int = 100,
    l1: float = L,
    l2: float = L,
) -> gf.Component:
    """Returns distributed bragg reflector.

    Args:
        w0: width.
        dw: delta width.
        n: number of elements.
        l1: length teeth1.
        l2: length teeth2.
    """
    c = gf.Component()

    # add_pins_left = partial(add_pins_siepic, prefix="o1")
    s = c << gf.components.straight(length=l1, cross_section=tech.strip_simple)
    _dbr = gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        cross_section=tech.strip_simple,
        decorator=None,
    )
    dbr = c << _dbr
    s.connect("o2", dbr.ports["o1"])
    c.add_port("o1", port=s.ports["o1"])
    c = add_pins_bbox_siepic(c)
    return c


@gf.cell
def dbr_cavity(**kwargs) -> gf.Component:
    d = dbr(**kwargs)
    return gf.components.cavity(component=d, coupler=coupler)


def dbr_cavity_te(component="dbr_cavity", **kwargs) -> gf.Component:
    component = gf.get_component(component, **kwargs)
    return add_fiber_array(component=component)


bend = gf.components.bend_euler
coupler = gf.components.coupler
coupler_ring = gf.components.coupler_ring
mmi1x2 = partial(gf.components.mmi1x2, cross_section=tech.strip_bbox)
coupler = partial(gf.components.coupler, cross_section=tech.strip_bbox)

ring_single = gf.partial(
    gf.components.ring_single,
    bend=bend,
    coupler_ring=coupler_ring,
    cross_section="strip",
)

spiral = gf.partial(gf.components.spiral_external_io)


@gf.cell
def ebeam_dc_halfring_straight(
    gap: float = 0.2,
    radius: float = 5.0,
    length_x: float = 4.0,
    cross_section="strip",
    siepic: bool = True,
    model: str = "ebeam_dc_halfring_straight",
    **kwargs,
) -> gf.Component:
    r"""Return a ring coupler.

    Args:
        gap: spacing between parallel coupled straight waveguides.
        radius: of the bends.
        length_x: length of the parallel coupled straight waveguides.
        cross_section: cross_section spec.
        siepic: if True adds siepic.
        kwargs: cross_section settings for bend and coupler.

    .. code::

           2             3
           |             |
            \           /
             \         /
           ---=========---
         1    length_x    4


    """
    c = gf.Component()
    coupler_ring = c << gf.components.coupler_ring(
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend_euler,
        cross_section=cross_section,
        **kwargs,
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
    return c


ebeam_dc_te1550 = partial(
    gf.components.coupler, decorator=add_pins_bbox_siepic_remove_layers
)
taper = partial(gf.components.taper)
spiral = partial(gf.components.spiral_external_io)
ring_with_crossing = partial(
    gf.components.ring_single_dut,
    component=ebeam_crossing4_2ports,
    port_name="o4",
    bend=bend_euler,
    cross_section=strip,
)


pad = partial(
    gf.components.pad,
    size=(75, 75),
    layer=LAYER.M2_ROUTER,
    bbox_layers=[LAYER.PAD_OPEN],
    bbox_offsets=[-1.8],
    decorator=add_pins_siepic_metal,
)


def add_label_electrical(component: Component, text: str, port_name: str = "e2"):
    """Adds labels for electrical port.

    Returns same component so it needs to be used as a decorator.
    """
    if port_name not in component.ports:
        raise ValueError(f"No port {port_name!r} in {list(component.ports.keys())}")

    component.add_label(
        text=text, position=component.ports[port_name].center, layer=LAYER.LABEL
    )
    return component


pad_array = gf.partial(gf.components.pad_array, pad=pad, spacing=(125, 125))
add_pads_rf = gf.partial(
    gf.routing.add_electrical_pads_top,
    component="ring_single_heater",
    pad_array=pad_array,
)
add_pads_dc = gf.partial(
    gf.routing.add_electrical_pads_top_dc,
    component="ring_single_heater",
    pad_array=pad_array,
)


@gf.cell
def add_fiber_array_pads_rf(
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
    add_label = gf.partial(add_label_electrical, text=text)
    rename_ports_and_add_label = gf.compose(
        add_label, gf.port.auto_rename_ports_electrical
    )
    c1 = add_pads_rf(component=c0, decorator=rename_ports_and_add_label)
    return add_fiber_array(component=c1, **kwargs)


@gf.cell
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
    add_label = gf.partial(add_label_electrical, text=text)
    return add_pads_rf(component=c0, decorator=add_label)


if __name__ == "__main__":
    # gf.clear_cache()
    # c = add_fiber_array(mmi1x2())
    # c = taper()

    # c = add_fiber_array_pads_rf()
    # c = add_fiber_array_pads_rf(c, optical_routing_type=2)

    # c = add_pads()
    # c = add_pads_rf()
    # c = coupler()
    # c = dbr(decorator=None)
    # c = dbr_cavity()
    # c = dbr_cavity_te()

    # c = ebeam_adiabatic_tm1550()
    # c = ebeam_bdc_te1550()
    # c = ebeam_crossing4()
    # c = ebeam_dc_halfring_straight()
    # c = ebeam_dc_te1550()
    # c = ebeam_y_1550()
    # c = ebeam_y_adiabatic_tapers()

    # c = gc_te1310()
    # c = gc_te1550()
    # c = gc_tm1550()
    # c = mmi1x2()
    # c = mzi(splitter='mmi1x2')
    c = mzi_heater()
    # c = pad()
    # c = ring_single()
    # c = ring_single_heater()
    # c = ring_with_crossing()
    # c = spiral()
    # c = thermal_phase_shifter0()
    # c = straight_one_pin()
    # c = ebeam_crossing4_2ports()
    c.show(show_ports=False)
