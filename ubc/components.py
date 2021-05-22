from typing import Callable, List, Optional, Tuple

import pp
from phidl import device_layout as pd
from phidl.device_layout import Label
from pp.add_labels import get_input_label
from pp.cell import cell
from pp.component import Component
from pp.components.ring_single_dut import ring_single_dut
from pp.port import Port, auto_rename_ports
from pp.rotate import rotate
from pp.tech import Factory
from pp.types import Layer
from pp.types import ComponentFactory, ComponentReference

from ubc.config import CONFIG
from ubc.import_gds import import_gds
from ubc.tech import LAYER
from ubc.add_pins import add_pins

L = 1.55 / 4 / 2 / 2.44


def get_optical_text(
    port: Port,
    gc: ComponentReference,
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
) -> str:
    """Return label for a component port and a grating coupler.

    Args:
        port: component port.
        gc: grating coupler reference.
        component_name: optional component name.
    """
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

    name = name.replace("_", "-")
    label = (
        f"opt_in_{polarization.upper()}_{int(wavelength_nm)}_device_"
        + f"{CONFIG.username}_({name})-{gc_index}-{port.name}"
    )
    return label


def get_input_labels_all(
    io_gratings,
    ordered_ports,
    component_name,
    layer_label=LAYER.LABEL,
    gc_port_name: str = "W0",
):
    """Return labels (elements list) for all component ports."""
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
    layer_label: Tuple[int, int] = LAYER.LABEL,
    gc_port_name: str = "W0",
    port_index: int = 1,
) -> List[Label]:
    """Return labels (elements list) for all component ports."""
    if port_index == -1:
        return get_input_labels_all(
            io_gratings=io_gratings,
            ordered_ports=ordered_ports,
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
        )
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_optical_text(
        port=port, gc=gc, gc_index=port_index, component_name=component_name
    )
    layer, texttype = pd._parse_layer(layer_label)
    label = pd.Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


@cell
def gc_te1550() -> Component:
    c = import_gds("ebeam_gc_te1550")
    c = rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1550
    auto_rename_ports(c)
    return c


@cell
def gc_te1550_broadband():
    c = import_gds("ebeam_gc_te1550_broadband")
    c = rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1550
    auto_rename_ports(c)
    return c


@cell
def gc_te1310():
    c = import_gds("ebeam_gc_te1310")
    c = rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1310
    auto_rename_ports(c)
    return c


@cell
def gc_tm1550():
    c = import_gds("ebeam_gc_tm1550")
    c = rotate(component=c, angle=180)
    c.polarization = "tm"
    c.wavelength = 1550
    auto_rename_ports(c)
    return c


@cell
def straight(
    length: float = 10.0,
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    with_pins: bool = True,
    **kwargs,
) -> Component:
    """Straight waveguide."""
    c = pp.components.straight(length=length, width=width, layer=layer, **kwargs)

    if with_pins:
        labels = [
            "Lumerical_INTERCONNECT_library=Design kits/EBeam",
            "Lumerical_INTERCONNECT_component=ebeam_wg_integral_1550",
            f"Spice_param:wg_width={width:.3f}u wg_length={length:.3f}u",
        ]

        for i, text in enumerate(labels):
            c.add_label(text=text, position=(length / 2, i * 0.1), layer=LAYER.DEVREC)
        add_pins(c)
    return c


@cell
def straight_no_pins(**kwargs):
    return straight(with_pins=False, **kwargs)


def crossing() -> Component:
    """TE waveguide crossing."""
    return import_gds("ebeam_crossing4", rename_ports=True)


def dc_broadband_te() -> Component:
    """Broadband directional coupler TE1550 50/50 power."""
    return import_gds("ebeam_bdc_te1550")


def dc_broadband_tm() -> Component:
    """Broadband directional coupler TM1550 50/50 power."""
    return import_gds("ebeam_bdc_tm1550")


def dc_adiabatic() -> Component:
    """Adiabatic directional coupler TE1550 50/50 power."""
    return import_gds("ebeam_adiabatic_te1550")


def y_adiabatic() -> Component:
    """Adiabatic Y junction TE1550 50/50 power."""
    return import_gds("ebeam_y_adiabatic")


def y_splitter() -> Component:
    """Y junction TE1550 50/50 power."""
    return import_gds("ebeam_y_1550")


def ring_with_crossing(**kwargs) -> Component:
    return ring_single_dut(component=crossing(), **kwargs)


def dbr(w0=0.5, dw=0.1, n=600, l1=L, l2=L) -> Component:
    return pp.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        straight=straight_no_pins,
    )


def dbr_cavity(**kwargs) -> Component:
    return pp.c.cavity(component=dbr(**kwargs))


def spiral(**kwargs):
    return pp.c.spiral_external_io(**kwargs)


@cell
def add_fiber_array(
    component: Component = straight,
    component_name: None = None,
    gc_port_name: str = "W0",
    get_input_labels_function: Callable = get_input_labels,
    with_align_ports: bool = False,
    optical_routing_type: int = 0,
    fanout_length: float = 0.0,
    grating_coupler: ComponentFactory = gc_tm1550,
    **kwargs,
) -> Component:
    """Returns component with grating couplers and labels on each port.

    Routes all component ports south.
    Can add align_ports loopback reference structure on the edges.

    Args:
        component: to connect
        component_name: for the label
        gc_port_name: grating coupler input port name 'W0'
        get_input_labels_function: function to get input labels for grating couplers
        with_align_ports: True, adds loopback structures
        optical_routing_type: None: autoselection, 0: no extension
        fanout_length: None  # if None, automatic calculation of fanout length
        taper_length: length of the taper
        grating_coupler: grating coupler instance, function or list of functions
        optical_io_spacing: SPACING_GC
    """

    c = pp.routing.add_fiber_array(
        component=component,
        component_name=component_name,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        get_input_labels_function=get_input_labels_function,
        with_align_ports=with_align_ports,
        optical_routing_type=optical_routing_type,
        layer_label=LAYER.LABEL,
        fanout_length=fanout_length,
        **kwargs,
    )
    c.rotate(-90)
    return c


COMPONENT_FACTORY = Factory()
COMPONENT_FACTORY.register(
    [
        add_fiber_array,
        crossing,
        dbr,
        dbr_cavity,
        dc_adiabatic,
        dc_broadband_te,
        dc_broadband_tm,
        gc_te1310,
        gc_te1550,
        gc_te1550_broadband,
        gc_tm1550,
        ring_with_crossing,
        spiral,
        straight,
        straight_no_pins,
        y_adiabatic,
        y_splitter,
    ]
)


__all__ = list(COMPONENT_FACTORY.factory.keys())


if __name__ == "__main__":
    # c = straight_no_pins()
    # c = add_fiber_array(component=c)
    # c = gc_tm1550()
    # print(c.ports.keys())
    c = add_fiber_array()
    c.show(show_ports=True)
