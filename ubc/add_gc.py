from typing import Callable, List, Optional, Tuple

import pp
from numpy import ndarray
from phidl import device_layout as pd
from phidl.device_layout import Label
from pp.add_labels import get_input_label
from pp.component import Component, ComponentReference
from pp.port import Port
from pp.rotate import rotate
from pp.routing.manhattan import round_corners
from ubc.bend90 import bend90
from ubc.config import conf
from ubc.import_gds import import_gds
from ubc.layers import LAYER
from ubc.waveguide import waveguide

gc_port_name = "W0"
layer_label = LAYER.LABEL


@pp.autoname
def gc_te1550() -> Component:
    c = import_gds("ebeam_gc_te1550")
    c = rotate(c, 180)
    c.polarization = "te"
    c.wavelength = 1550
    return c


def gc_te1550_broadband():
    c = import_gds("ebeam_gc_te1550_broadband")
    return c


def gc_te1310():
    c = import_gds("ebeam_gc_te1310")
    c.polarization = "te"
    c.wavelength = 1310
    return c


def gc_tm1550():
    c = import_gds("ebeam_gc_tm1550")
    c.polarization = "tm"
    c.wavelength = 1550
    return c


def connect_strip(
    way_points: ndarray = [],
    bend_factory: Component = bend90(),
    straight_factory: Callable = waveguide,
    bend_radius: float = 10.0,
    wg_width: float = 0.5,
    **kwargs,
) -> ComponentReference:
    """
    Returns a deep-etched route formed by the given way_points with
    bends instead of corners and optionally tapers in straight sections.
    """
    bend90 = pp.call_if_func(bend_factory, radius=bend_radius, width=wg_width)
    connector = round_corners(way_points, bend90, straight_factory)
    return connector


@pp.autoname
def taper_factory(layer=LAYER.WG, layers_cladding=[], **kwargs):
    c = pp.c.taper(layer=layer, layers_cladding=layers_cladding, **kwargs)
    return c


def get_optical_text(
    port: Port,
    gc: ComponentReference,
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
) -> str:
    polarization = gc.get_property("polarization")
    wavelength_nm = gc.get_property("wavelength")

    assert polarization.upper() in [
        "TE",
        "TM",
    ], f"Not valid polarization {polarization.upper()} in [TE, TM]"
    assert (
        isinstance(wavelength_nm, (int, float)) and 1000 < wavelength_nm < 2000
    ), f"{wavelength_nm} is Not valid 1000 < wavelength < 2000"

    if component_name:
        name = component_name

    elif type(port.parent) == pp.Component:
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    name += f"_{port.name}"
    name = name.replace("_", "-")
    label = f"opt_in_{polarization.upper()}_{int(wavelength_nm)}_device_{conf.username}_{name}"
    return label


def get_input_labels_all(
    io_gratings,
    ordered_ports,
    component_name,
    layer_label=layer_label,
    gc_port_name=gc_port_name,
):
    """ get labels for all component ports """
    elements = []
    for i, g in enumerate(io_gratings):
        label = get_input_label(
            port=ordered_ports[i],
            gc=g,
            gc_index=i,
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
        )
        elements += [label]

    return elements


def get_input_labels(
    io_gratings: List[ComponentReference],
    ordered_ports: List[Port],
    component_name: str,
    layer_label: Tuple[int, int] = layer_label,
    gc_port_name: str = gc_port_name,
    port_index: int = 1,
) -> List[Label]:
    """ get labels for all component ports """
    if port_index == -1:
        return get_input_labels_all(
            io_gratings=io_gratings,
            ordered_ports=ordered_ports,
            component_name=component_name,
            gc_port_name=gc_port_name,
            port_index=port_index,
        )
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_optical_text(
        port=port, gc=gc, gc_index=port_index, component_name=component_name
    )
    print(text)
    layer, texttype = pd._parse_layer(layer_label)
    label = pd.Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


def add_gc(
    component: Component = waveguide,
    component_name: None = None,
    layer_label: Tuple[int, int] = LAYER.LABEL,
    grating_coupler: Callable = gc_te1550,
    bend_factory: Callable = bend90,
    straight_factory: Callable = waveguide,
    taper_factory: Callable = taper_factory,
    route_filter: Callable = connect_strip,
    gc_port_name: str = "W0",
    get_input_labels_function: Callable = get_input_labels,
    with_align_ports: bool = False,
    optical_routing_type: int = 0,
    fanout_length: int = 0,
    **kwargs,
) -> Component:
    c = pp.routing.add_fiber_array(
        component=component,
        component_name=component_name,
        bend_factory=bend_factory,
        straight_factory=straight_factory,
        route_filter=route_filter,
        grating_coupler=grating_coupler,
        layer_label=layer_label,
        taper_factory=taper_factory,
        gc_port_name=gc_port_name,
        get_input_labels_function=get_input_labels_function,
        with_align_ports=with_align_ports,
        optical_routing_type=optical_routing_type,
        fanout_length=fanout_length,
        **kwargs,
    )
    c = rotate(c, -90)
    return c


if __name__ == "__main__":
    import ubc

    # c = gc_te1550()
    # print(c.ports)
    c = add_gc(component=ubc.mzi(delta_length=100))
    # c = add_gc(component=waveguide())
    pp.show(c)
    pp.write_gds(c, "mzi.gds")
