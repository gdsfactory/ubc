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


pack_doe = gf.c.pack_doe
pack_doe_grid = gf.c.pack_doe_grid

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
    c = add_pads()
    c.pprint_ports()
    c.show()
