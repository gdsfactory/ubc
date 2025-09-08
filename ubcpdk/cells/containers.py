"""This module contains cells that contain other cells."""

from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component, ComponentReference
from gdsfactory.typings import (
    AngleInDegrees,
    CellSpec,
    ComponentSpec,
    CrossSectionSpec,
    Float2,
    LayerSpec,
)

from ubcpdk.config import CONFIG
from ubcpdk.tech import LAYER

gc = "ebeam_gc_te1550"


def clean_name(name: str) -> str:
    return name.replace("_", ".")


def add_label_electrical(component: Component, text: str, port_name: str = "e2"):
    """Adds labels for electrical port.

    Returns same component so it needs to be used as a decorator.
    """
    if port_name not in component.ports:
        port_names = [port.name for port in component.ports]
        raise ValueError(f"No port {port_name!r} in {port_names}")

    component.add_label(
        text=text, position=component.ports[port_name].dcenter, layer=LAYER.TEXT
    )
    return component


def get_input_label_text(
    gc: ComponentReference,
    component_name: str | None = None,
    username: str = CONFIG.username,
) -> str:
    """Return label for port and a grating coupler.

    Args:
        gc: grating coupler reference.
        component_name: optional component name.
        username: for the label.
    """
    polarization = gc.info.get("polarization")
    wavelength = gc.info.get("wavelength")

    gc_cell_name = gc.name.lower()
    if polarization is None:
        if "te" in gc_cell_name:
            polarization = "te"
        else:
            polarization = "tm"

    if wavelength is None:
        if "1310" in gc_cell_name:
            wavelength = 1.310
        else:
            wavelength = 1.550

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


@gf.cell
def add_fiber_array(
    component: ComponentSpec = "ring_single",
    component_name: str | None = None,
    gc_port_name: str = "o1",
    with_loopback: bool = False,
    fanout_length: float | None = 0,
    grating_coupler: ComponentSpec = gc,
    cross_section: CrossSectionSpec = "strip",
    straight: ComponentSpec = "straight",
    taper: ComponentSpec | None = None,
    mirror_grating_coupler: bool = True,
    gc_rotation: float = +90,
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
        fanout_length: None  # if None, automatic calculation of fanout length.
        grating_coupler: grating coupler instance, function or list of functions.
        cross_section: spec.
        straight: straight component.
        taper: taper component.
        kwargs: cross_section settings.

    """
    c = gf.Component()
    component = gf.get_component(component)

    component_with_grating_coupler = gf.routing.add_fiber_array(
        straight=straight,
        bend="bend_euler",
        component=component,
        component_name=component_name,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        with_loopback=with_loopback,
        fanout_length=fanout_length,
        cross_section=cross_section,
        taper=taper,
        mirror_grating_coupler=mirror_grating_coupler,
        gc_rotation=gc_rotation,
        **kwargs,
    )
    component_with_grating_coupler.name = component.name + "_with_gc"
    ref = c << component_with_grating_coupler
    ref.rotate(-90)
    c.add_ports(ref.ports)
    c.copy_child_info(component)

    component_name = component_name or component.name
    grating_coupler = gf.get_component(grating_coupler)
    label = get_input_label_text(gc=grating_coupler, component_name=component_name)
    c.add_label(position=c.ports["o1"].dcenter, text=label, layer=LAYER.TEXT)
    return c


# @gf.cell
# def add_fiber_array_bottom(
#     component: ComponentSpec = "straight",
#     grating_coupler=gc,
#     gc_port_name: str = "o1",
#     component_name: str | None = None,
#     cross_section: CrossSectionSpec = "strip",
#     gc_rotation: float = +90,
#     radius_loopback: float = 10,
#     mirror_grating_coupler: bool = True,
#     **kwargs,
# ) -> Component:
#     """Returns component with south routes and grating_couplers.

#     You can also use pads or other terminations instead of grating couplers.

#     Args:
#         component: component spec to connect to grating couplers.
#         grating_coupler: spec for route terminations.
#         gc_port_name: grating coupler input port name.
#         component_name: optional for the label.
#         cross_section: cross_section function.
#         gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
#         radius_loopback: optional radius of the loopback bend. Defaults to the cross_section.
#         kwargs: additional arguments.

#     Keyword Args:
#         bend: bend spec.
#         straight: straight spec.
#         fanout_length: if None, automatic calculation of fanout length.
#         max_y0_optical: in um.
#         with_loopback: True, adds loopback structures.
#         with_loopback_inside: True, adds loopback structures inside the component.
#         straight_separation: from edge to edge.
#         list_port_labels: None, adds TM labels to port indices in this list.
#         connected_port_list_ids: names of ports only for type 0 optical routing.
#         nb_optical_ports_lines: number of grating coupler lines.
#         force_manhattan: False
#         excluded_ports: list of port names to exclude when adding gratings.
#         grating_indices: list of grating coupler indices.
#         routing_straight: function to route.
#         routing_method: route_single.
#         optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
#         input_port_indexes: to connect.
#         pitch: in um.
#         radius: optional radius of the bend. Defaults to the cross_section.
#         route_backwards: route from component to grating coupler or vice-versa.

#     .. plot::
#         :include-source:

#         import gdsfactory as gf

#         c = gf.components.crossing()
#         cc = gf.routing.add_fiber_array(
#             component=c,
#             optical_routing_type=2,
#             grating_coupler=gf.components.grating_coupler_elliptical_te,
#             with_loopback=False
#         )
#         cc.plot()

#     """
#     return gf.routing.add_fiber_array(
#         component=component,
#         grating_coupler=grating_coupler,
#         gc_port_name=gc_port_name,
#         gc_rotation=gc_rotation,
#         component_name=component_name,
#         cross_section=cross_section,
#         radius_loopback=radius_loopback,
#         mirror_grating_coupler=mirror_grating_coupler,
#         **kwargs,
#     )


# @gf.cell
# def add_fiber_single(
#     component: ComponentSpec = "straight",
#     grating_coupler=gc,
#     gc_port_name: str = "o1",
#     component_name: str | None = None,
#     cross_section: CrossSectionSpec = "strip",
#     taper: ComponentSpec | None = None,
#     input_port_names: list[str] | tuple[str, ...] | None = None,
#     pitch: float = 70,
#     with_loopback: bool = True,
#     loopback_spacing: float = 100.0,
#     **kwargs,
# ) -> Component:
#     """Returns component with south routes and grating_couplers.

#     You can also use pads or other terminations instead of grating couplers.

#     Args:
#         component: component spec to connect to grating couplers.
#         grating_coupler: spec for route terminations.
#         gc_port_name: grating coupler input port name.
#         component_name: optional for the label.
#         cross_section: cross_section function.
#         taper: taper spec.
#         input_port_names: list of input port names to connect to grating couplers.
#         pitch: spacing between fibers.
#         with_loopback: adds loopback structures.
#         loopback_spacing: spacing between loopback and test structure.
#         kwargs: additional arguments.

#     Keyword Args:
#         bend: bend spec.
#         straight: straight spec.
#         fanout_length: if None, automatic calculation of fanout length.
#         max_y0_optical: in um.
#         with_loopback: True, adds loopback structures.
#         straight_separation: from edge to edge.
#         list_port_labels: None, adds TM labels to port indices in this list.
#         connected_port_list_ids: names of ports only for type 0 optical routing.
#         nb_optical_ports_lines: number of grating coupler lines.
#         force_manhattan: False
#         excluded_ports: list of port names to exclude when adding gratings.
#         grating_indices: list of grating coupler indices.
#         routing_straight: function to route.
#         routing_method: route_single.
#         optical_routing_type: None: auto, 0: no extension, 1: standard, 2: check.
#         gc_rotation: fiber coupler rotation in degrees. Defaults to -90.
#         input_port_indexes: to connect.

#     .. plot::
#         :include-source:

#         import gdsfactory as gf

#         c = gf.components.crossing()
#         cc = gf.routing.add_fiber_array(
#             component=c,
#             optical_routing_type=2,
#             grating_coupler=gf.components.grating_coupler_elliptical_te,
#             with_loopback=False
#         )
#         cc.plot()

#     """
#     return gf.routing.add_fiber_single(
#         component=component,
#         grating_coupler=grating_coupler,
#         gc_port_name=gc_port_name,
#         component_name=component_name,
#         cross_section=cross_section,
#         taper=taper,
#         input_port_names=input_port_names,
#         pitch=pitch,
#         with_loopback=with_loopback,
#         loopback_spacing=loopback_spacing,
#         **kwargs,
#     )


# @gf.cell
# def add_pads_top(
#     component: ComponentSpec = "straight_metal",
#     port_names: Strs | None = None,
#     component_name: str | None = None,
#     cross_section: CrossSectionSpec = "metal_routing",
#     pad_port_name: str = "e1",
#     pad: ComponentSpec = "pad",
#     bend: ComponentSpec = "wire_corner",
#     straight_separation: float = 15.0,
#     pad_pitch: float = 100.0,
#     taper: ComponentSpec | None = None,
#     port_type: str = "electrical",
#     allow_width_mismatch: bool = True,
#     fanout_length: float | None = 80,
#     route_width: float | list[float] | None = 0,
#     **kwargs,
# ) -> Component:
#     """Returns new component with ports connected top pads.

#     Args:
#         component: component spec to connect to.
#         port_names: list of port names to connect to pads.
#         component_name: optional for the label.
#         cross_section: cross_section function.
#         pad_port_name: pad port name.
#         pad: pad function.
#         bend: bend function.
#         straight_separation: from edge to edge.
#         pad_pitch: spacing between pads.
#         taper: taper function.
#         port_type: port type.
#         allow_width_mismatch: if True, allows width mismatch.
#         fanout_length: length of the fanout.
#         route_width: width of the route.
#         kwargs: additional arguments.

#     .. plot::
#         :include-source:

#         import gdsfactory as gf
#         c = gf.c.nxn(
#             xsize=600,
#             ysize=200,
#             north=2,
#             south=3,
#             wg_width=10,
#             layer="M3",
#             port_type="electrical",
#         )
#         cc = gf.routing.add_pads_top(component=c, port_names=("e1", "e4"), fanout_length=50)
#         cc.plot()

#     """
#     return gf.routing.add_pads_top(
#         component=component,
#         port_names=port_names,
#         component_name=component_name,
#         cross_section=cross_section,
#         pad_port_name=pad_port_name,
#         pad=pad,
#         bend=bend,
#         straight_separation=straight_separation,
#         pad_pitch=pad_pitch,
#         taper=taper,
#         port_type=port_type,
#         allow_width_mismatch=allow_width_mismatch,
#         fanout_length=fanout_length,
#         route_width=route_width,
#         **kwargs,
#     )


@gf.cell
def pack_doe(
    doe: ComponentSpec,
    settings: dict[str, tuple[Any, ...]],
    do_permutations: bool = False,
    function: CellSpec | None = None,
    **kwargs,
) -> Component:
    """Packs a component DOE (Design of Experiment) using pack.

    Args:
        doe: function to return Components.
        settings: component settings.
        do_permutations: for each setting.
        function: to apply (add padding, grating couplers).
        kwargs: for pack.

    Keyword Args:
        spacing: Minimum distance between adjacent shapes.
        aspect_ratio: (width, height) ratio of the rectangular bin.
        max_size: Limits the size into which the shapes will be packed.
        sort_by_area: Pre-sorts the shapes by area.
        density: Values closer to 1 pack tighter but require more computation.
        precision: Desired precision for rounding vertex coordinates.
        text: Optional function to add text labels.
        text_prefix: for labels. For example. 'A' for 'A1', 'A2'...
        text_offsets: relative to component size info anchor. Defaults to center.
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc).
        name_prefix: for each packed component (avoids the Unnamed cells warning).
            Note that the suffix contains a uuid so the name will not be deterministic.
        rotation: for each component in degrees.
        h_mirror: horizontal mirror in y axis (x, 1) (1, 0). This is the most common.
        v_mirror: vertical mirror using x axis (1, y) (0, y).
    """
    return gf.components.pack_doe(
        doe=doe,
        settings=settings,
        do_permutations=do_permutations,
        function=function,
        **kwargs,
    )


@gf.cell
def pack_doe_grid(
    doe: ComponentSpec,
    settings: dict[str, tuple[Any, ...]],
    do_permutations: bool = False,
    function: CellSpec | None = None,
    with_text: bool = False,
    **kwargs,
) -> Component:
    """Packs a component DOE (Design of Experiment) using grid.

    Args:
        doe: function to return Components.
        settings: component settings.
        do_permutations: for each setting.
        function: to apply to component (add padding, grating couplers).
        with_text: includes text label.
        kwargs: for grid.

    Keyword Args:
        spacing: between adjacent elements on the grid, can be a tuple for
            different distances in height and width.
        separation: If True, guarantees elements are separated with fixed spacing
            if False, elements are spaced evenly along a grid.
        shape: x, y shape of the grid (see np.reshape).
            If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: {'x', 'xmin', 'xmax'} for x (column) alignment along.
        align_y: {'y', 'ymin', 'ymax'} for y (row) alignment along.
        edge_x: {'x', 'xmin', 'xmax'} for x (column) (ignored if separation = True).
        edge_y: {'y', 'ymin', 'ymax'} for y (row) (ignored if separation = True).
        rotation: for each component in degrees.
        h_mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.
        v_mirror: vertical mirror using x axis (1, y) (0, y).
    """
    return gf.components.pack_doe_grid(
        doe=doe,
        settings=settings,
        do_permutations=do_permutations,
        function=function,
        with_text=with_text,
        **kwargs,
    )


@gf.cell
def add_fiber_array_pads_rf(
    component: ComponentSpec = "ring_single_heater",
    username: str = CONFIG.username,
    orientation: float = 0,
    pad_yspacing: float = 50,
    component_name: str | None = None,
    **kwargs,
) -> Component:
    """Returns fiber array with label and electrical pads.

    Args:
        component: to add fiber array and pads.
        username: for the label.
        orientation: for adding pads.
        pad_yspacing: for adding pads.
        component_name: for the label.
        kwargs: for add_fiber_array.
    """

    c0 = gf.get_component(component)
    component_name = component_name or c0.name
    component_name = clean_name(component_name)
    text = f"elec_{username}-{component_name}_G"
    c1 = add_pads_rf(component=c0, orientation=orientation, spacing=(0, pad_yspacing))
    c1.name = text

    add_label_electrical(component=c1, text=text)
    return add_fiber_array(component=c1, component_name=component_name, **kwargs)


@gf.cell
def pad_array(
    pad: ComponentSpec = "pad",
    columns: int = 6,
    rows: int = 1,
    column_pitch: float = 125.0,
    row_pitch: float = 125.0,
    port_orientation: AngleInDegrees = 270,
    size: Float2 | None = None,
    layer: LayerSpec | None = None,
    centered_ports: bool = False,
    auto_rename_ports: bool = False,
) -> gf.Component:
    """Returns 2D array of pads.

    Args:
        pad: pad element.
        columns: number of columns.
        rows: number of rows.
        column_pitch: x pitch.
        row_pitch: y pitch.
        port_orientation: port orientation in deg. None for low speed DC ports.
        size: pad size.
        layer: pad layer.
        centered_ports: True add ports to center. False add ports to the edge.
        auto_rename_ports: True to auto rename ports.
    """
    return gf.c.pad_array(
        pad=pad,
        columns=columns,
        rows=rows,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
        port_orientation=port_orientation,
        size=size,
        layer=layer,
        centered_ports=centered_ports,
        auto_rename_ports=auto_rename_ports,
    )


add_pads_rf = partial(
    gf.routing.add_electrical_pads_top,
    component="ring_single_heater",
    pad_array="pad_array",
)


@gf.cell
def add_pads(
    component: ComponentSpec = "ring_single_heater",
    username: str = CONFIG.username,
    label_port_name="l_e1",
    **kwargs,
) -> Component:
    """Returns fiber array with label and electrical pads.

    Args:
        component: to add fiber array and pads.
        username: for the label.
        kwargs: for add_fiber_array.
    """
    c0 = gf.get_component(component).copy()
    text = f"elec_{username}-{clean_name(c0.name)}_G"
    c0 = add_label_electrical(c0, text=text, port_name=label_port_name)
    return add_pads_rf(component=c0, **kwargs)


if __name__ == "__main__":
    from ubcpdk import PDK

    PDK.activate()

    # c = add_fiber_array("ring_double")
    # c =gf.get_component(gc)
    # c = pack_doe()
    c = add_pads_rf()
    c.pprint_ports()
    c.show()
