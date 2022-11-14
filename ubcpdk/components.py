"""Cells imported from the PDK."""

import gdsfactory as gf
from gdsfactory.types import ComponentSpec

from ubcpdk.import_gds import import_gds, import_gc
from ubcpdk.tech import strip, LAYER_STACK, LAYER


um = 1e-6

straight = gf.partial(
    gf.components.straight,
    cross_section="strip",
)
bend_euler = gf.partial(
    gf.components.bend_euler,
    cross_section="strip",
)
bend_s = gf.partial(
    gf.components.bend_s,
    cross_section="strip",
)

info1550te = dict(polarization="te", wavelength=1.55)
info1310te = dict(polarization="te", wavelength=1.31)
info1550tm = dict(polarization="tm", wavelength=1.55)
info1310tm = dict(polarization="tm", wavelength=1.31)


# @gf.cell
# def Packaging_FibreArray_8ch() -> gf.Component:
#     """Return Packaging_FibreArray_8ch fixed cell."""
#     return import_gds("Packaging_FibreArray_8ch.gds")
# @gf.cell
# def thermal_phase_shifters() -> gf.Component:
#     """Return thermal_phase_shifters fixed cell."""
#     return import_gds("thermal_phase_shifters.gds")


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
def ebeam_gc_te1310() -> gf.Component:
    """Return ebeam_gc_te1310 fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_gc_te1310()
      c.plot()
    """
    return import_gc("ebeam_gc_te1310.gds", info=info1310te)


@gf.cell
def ebeam_gc_te1310_8deg() -> gf.Component:
    """Return ebeam_gc_te1310_8deg fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_gc_te1310_8deg()
      c.plot()
    """
    return import_gc("ebeam_gc_te1310_8deg.gds", info=info1310te)


@gf.cell
def ebeam_gc_te1310_broadband() -> gf.Component:
    """Return ebeam_gc_te1310_broadband fixed cell.

    .. plot::
      :include-source:

      import ubcpdk

      c = ubcpdk.components.ebeam_gc_te1310_broadband()
      c.plot()
    """
    return import_gc("ebeam_gc_te1310_broadband.gds", info=info1310te)


@gf.cell
def ebeam_gc_te1550() -> gf.Component:
    """Return ebeam_gc_te1550 fixed cell."""
    return import_gc("ebeam_gc_te1550.gds", info=info1550te)


@gf.cell
def ebeam_gc_te1550_90nmSlab() -> gf.Component:
    """Return ebeam_gc_te1550_90nmSlab fixed cell."""
    return import_gc("ebeam_gc_te1550_90nmSlab.gds", info=info1550te)


@gf.cell
def ebeam_gc_te1550_broadband() -> gf.Component:
    """Return ebeam_gc_te1550_broadband fixed cell."""
    return import_gc("ebeam_gc_te1550_broadband.gds", info=info1550te)


@gf.cell
def ebeam_gc_tm1550() -> gf.Component:
    """Return ebeam_gc_tm1550 fixed cell."""
    return import_gc("ebeam_gc_tm1550.gds", info=info1550tm)


mzi = gf.partial(
    gf.components.mzi,
    splitter=ebeam_y_1550,
    bend=bend_euler,
    straight=straight,
    port_e1_splitter="opt2",
    port_e0_splitter="opt3",
    port_e1_combiner="opt2",
    port_e0_combiner="opt3",
    cross_section="strip",
)


@gf.cell
def add_fiber_array(
    component: ComponentSpec = "straight",
    gc_port_name: str = "opt1",
    grating_coupler: ComponentSpec = ebeam_gc_te1550,
    **kwargs
) -> gf.Component:
    """Returns component with grating couplers and labels on each port.

    Routes all component ports south.
    Can add align_ports loopback reference structure on the edges.

    Args:
        component: to connect.
        gc_port_name: grating coupler input port name 'o1'.
        grating_coupler: grating coupler instance, function or list of functions.

    keyword Args:
        component_name: for the label.
        get_input_labels_function: function to get input labels for grating couplers.
        with_loopback: True, adds loopback structures.
        optical_routing_type: None: autoselection, 0: no extension.
        fanout_length: None  # if None, automatic calculation of fanout length.
        cross_section: spec.
        layer_label: for label.

    """
    from gdsfactory.labels.siepic import add_fiber_array_siepic

    return add_fiber_array_siepic(
        component=component,
        gc_port_name=gc_port_name,
        grating_coupler=grating_coupler,
        **kwargs
    )


L = 1.55 / 4 / 2 / 2.44


@gf.cell
def dbr(
    w0: float = 0.5,
    dw: float = 0.1,
    n: int = 600,
    l1: float = L,
    l2: float = L,
    cross_section="strip_no_pins",
) -> gf.Component:

    return gf.components.dbr(
        w1=w0 - dw / 2, w2=w0 + dw / 2, n=n, l1=l1, l2=l2, cross_section=cross_section
    )


def dbr_cavity(**kwargs) -> gf.Component:
    return gf.components.cavity(component=dbr(**kwargs), coupler=coupler)


def dbr_cavity_te(component="dbr_cavity", **kwargs) -> gf.Component:
    component = gf.get_component(component, **kwargs)
    return add_fiber_array(component=component)


bend = gf.components.bend_euler
coupler = gf.components.coupler
coupler_ring = gf.components.coupler_ring

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
    **kwargs
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
    return c


ebeam_dc_te1550 = gf.components.coupler
spiral = gf.partial(gf.components.spiral_external_io)
ring_with_crossing = gf.partial(
    gf.components.ring_single_dut,
    component=ebeam_crossing4,
    port_name="opt4",
    bend=bend_euler,
    straight=straight,
    cross_section=strip,
)


if __name__ == "__main__":
    # c = ebeam_crossing4()
    # c = ebeam_dc_te1550()
    # c = ebeam_y_1550()
    # c = ebeam_bdc_te1550()
    c = ebeam_gc_te1550()
    # c = spiral()
    # c= coupler()
    # c = ebeam_gc_tm1550()
    # c = add_fiber_array()
    # c = dbr_cavity_te()
    c.show(show_ports=True)
