"""Technology definitions.

- LayerStack
- cross_sections (xs_)
- constants (WIDTH, CLADDING_OFFSET ...)
"""
import sys
from functools import partial

from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_section_factories
from gdsfactory.technology import LayerStack, LayerLevel
from gdsfactory.typings import Layer, LayerSpec, Callable, LayerSpecs, Optional
from gdsfactory.add_pins import add_pin_path
from gdsfactory.component import Component

from ubcpdk.config import PATH

nm = 1e-3
pin_length = 10 * nm


class LayerMapUbc(BaseModel):
    WG: Layer = (1, 0)
    WG2: Layer = (31, 0)
    M1_HEATER: Layer = (11, 0)
    M2_ROUTER: Layer = (12, 0)
    MTOP: Layer = (12, 0)
    PAD_OPEN: Layer = (13, 0)

    DEVREC: Layer = (68, 0)
    LABEL: Layer = (10, 0)
    PORT: Layer = (1, 10)  # PinRec
    PORTE: Layer = (1, 11)  # PinRecM
    FLOORPLAN: Layer = (99, 0)

    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (10, 0)
    LABEL_INSTANCE: Layer = (66, 0)
    SHOW_PORTS: Layer = (1, 13)
    PADDING: Layer = (67, 0)
    SLAB150: Layer = (2, 0)
    WAFER: Layer = (99999, 0)

    class Config:
        frozen = True
        extra = "forbid"


LAYER = LayerMapUbc()


def add_labels_to_ports_optical(
    component: Component,
    label_layer: LayerSpec = LAYER.TEXT,
    port_type: Optional[str] = "optical",
    **kwargs,
) -> Component:
    """Add labels to component ports.

    Args:
        component: to add labels.
        label_layer: layer spec for the label.
        port_type: to select ports.

    keyword Args:
        layer: select ports with GDS layer.
        prefix: select ports with prefix in port name.
        orientation: select ports with orientation in degrees.
        width: select ports with port width.
        layers_excluded: List of layers to exclude.
        port_type: select ports with port_type (optical, electrical, vertical_te).
        clockwise: if True, sort ports clockwise, False: counter-clockwise.
    """
    suffix = "o3_0" if len(component.ports) == 4 else "o2_0"
    ports = component.get_ports_list(port_type=port_type, suffix=suffix, **kwargs)
    for port in ports:
        component.add_label(text=port.name, position=port.center, layer=label_layer)

    return component


def add_bbox_siepic(
    component: Component,
    bbox_layer: LayerSpec = "DEVREC",
    remove_layers: LayerSpecs = ("PORT", "PORTE"),
) -> Component:
    """Add bounding box device recognition layer.

    Args:
        component: to add bbox.
        bbox_layer: bounding box.
        remove_layers: remove other layers.
    """
    from gdsfactory.pdk import get_layer

    bbox_layer = get_layer(bbox_layer)
    remove_layers = remove_layers or []
    remove_layers = list(remove_layers) + [bbox_layer]
    remove_layers = [get_layer(layer) for layer in remove_layers]
    component = component.remove_layers(layers=remove_layers, recursive=False)

    if bbox_layer:
        component.add_padding(default=0, layers=(bbox_layer,))
    return component


def add_pins_siepic(
    component: Component,
    function: Callable = add_pin_path,
    port_type: str = "optical",
    layer_pin: LayerSpec = "PORT",
    pin_length: float = pin_length,
    **kwargs,
) -> Component:
    """Add pins.

    Enables you to run SiEPIC verification tools:
    To Run verification install SiEPIC-tools KLayout package
    then hit V shortcut in KLayout to run verification

    - ensure no disconnected pins
    - netlist extraction

    Args:
        component: to add pins.
        function: to add pin.
        port_type: optical, electrical, ...
        layer_pin: pin layer.
        pin_length: length of the pin marker for the port.

    Keyword Args:
        layer: select ports with GDS layer.
        prefix: select ports with port name.
        orientation: select ports with orientation in degrees.
        width: select ports with port width.
        layers_excluded: List of layers to exclude.
        port_type: select ports with port_type (optical, electrical, vertical_te).
        clockwise: if True, sort ports clockwise, False: counter-clockwise.
    """
    for p in component.get_ports_list(port_type=port_type, **kwargs):
        function(component=component, port=p, layer=layer_pin, pin_length=pin_length)

    return component


add_pins_siepic_metal = partial(
    add_pins_siepic, port_type="placement", layer_pin=LAYER.PORTE
)


def add_pins_bbox_siepic(
    component: Component,
    function: Callable = add_pin_path,
    port_type: str = "optical",
    layer_pin: Layer = LAYER.PORT,
    pin_length: float = pin_length,
    bbox_layer: Layer = LAYER.DEVREC,
    padding: float = 0,
    remove_layers: bool = False,
) -> Component:
    """Add bounding box device recognition layer.

    Args:
        component: to add pins.
        function: to add pins.
        port_type: optical, electrical...
        layer_pin: for pin.
        pin_length: in um.
        bbox_layer: bounding box layer.
        padding: around device.
        remove_layers: removes old layers.
    """
    c = component
    if remove_layers or component.name.startswith(("mmi", "dbr")):
        remove_layers = (layer_pin, bbox_layer, "TEXT")
        c = component.remove_layers(layers=remove_layers)

    if bbox_layer not in c.layers:
        c.add_padding(default=padding, layers=(bbox_layer,))

    if layer_pin not in c.layers:
        c = add_pins_siepic(
            component=component,
            function=function,
            port_type=port_type,
            layer_pin=layer_pin,
            pin_length=pin_length,
        )
    return c


add_pins_bbox_siepic_remove_layers = partial(add_pins_bbox_siepic, remove_layers=True)


add_pins_bbox_siepic_metal = partial(
    add_pins_bbox_siepic, port_type="placement", layer_pin=LAYER.PORTE
)


def get_layer_stack(
    thickness_wg: float = 220 * nm,
    zmin_heater: float = 1.1,
    thickness_heater: float = 700 * nm,
    thickness_metal2: float = 700 * nm,
    substrate_thickness: float = 10.0,
    box_thickness: float = 3.0,
) -> LayerStack:
    """Returns generic LayerStack.

    based on paper https://www.degruyter.com/document/doi/10.1515/nanoph-2013-0034/html

    Args:
        thickness_wg: waveguide thickness in um.
        zmin_heater: TiN heater.
        thickness_heater: TiN thickness.
        zmin_metal2: metal2.
        thickness_metal2: metal2 thickness.
        substrate_thickness: substrate thickness in um.
        box_thickness: bottom oxide thickness in um.
    """

    class GenericLayerStack(LayerStack):
        substrate = LayerLevel(
            layer=LAYER.WAFER,
            thickness=substrate_thickness,
            zmin=-substrate_thickness - box_thickness,
            material="si",
            info={"mesh_order": 99},
        )
        box = LayerLevel(
            layer=LAYER.WAFER,
            thickness=box_thickness,
            zmin=-box_thickness,
            material="sio2",
            info={"mesh_order": 99},
        )
        clad = LayerLevel(
            layer=LAYER.WAFER,
            thickness=zmin_heater + thickness_heater,
            zmin=0,
            material="sio2",
            info={"mesh_order": 100},
        )
        core = LayerLevel(
            layer=LAYER.WG,
            thickness=thickness_wg,
            zmin=0.0,
            material="si",
            info={"mesh_order": 1},
            sidewall_angle=10,
            width_to_z=0.5,
        )
        core2 = LayerLevel(
            layer=LAYER.WG2,
            thickness=thickness_wg,
            zmin=0.0,
            material="si",
            info={"mesh_order": 1},
            sidewall_angle=10,
            width_to_z=0.5,
        )
        heater = LayerLevel(
            layer=LAYER.M1_HEATER,
            thickness=750e-3,
            zmin=zmin_heater,
            material="TiN",
            info={"mesh_order": 1},
        )
        metal2 = LayerLevel(
            layer=LAYER.M2_ROUTER,
            thickness=thickness_metal2,
            zmin=zmin_heater + thickness_heater,
            material="Aluminum",
            info={"mesh_order": 2},
        )

    return GenericLayerStack()


class Tech(BaseModel):
    name: str = "ubc"
    layer: LayerMapUbc = LAYER

    fiber_array_spacing: float = 250.0
    WG = {"width": 0.5}
    DEVREC = {"width": 0.5}


TECH = Tech()
LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews.from_lyp(PATH.lyp)


strip_wg_simulation_info = dict(
    model="ebeam_wg_integral_1550",
    layout_model_property_pairs=dict(
        # interconnect_property_name=(layout_property_name, scaling_value)
        wg_length=("length", 1e-6),
        wg_width=("width", 1e-6),
    ),
    layout_model_port_pairs=dict(o1="port 1", o2="port 2"),
    properties=dict(annotate=False),
)

cladding_layers_optical_siepic = ("DEVREC",)  # for SiEPIC verification
cladding_offsets_optical_siepic = (0,)  # for SiEPIC verification

strip = partial(
    gf.cross_section.cross_section,
    add_pins=add_pins_siepic,
    # add_bbox=add_bbox_siepic,
    cladding_layers=cladding_layers_optical_siepic,
    cladding_offsets=cladding_offsets_optical_siepic,
    # bbox_layers=cladding_layers_optical_siepic,
    # bbox_offsets=cladding_offsets_optical_siepic,
)
strip_heater_metal = partial(
    gf.cross_section.strip_heater_metal,
    layer="WG",
    heater_width=2.5,
    layer_heater=LAYER.M1_HEATER,
    cladding_layers=cladding_layers_optical_siepic,
    cladding_offsets=cladding_offsets_optical_siepic,
    add_pins=add_pins_siepic,
)

strip_simple = gf.cross_section.cross_section
strip_bbox_only = gf.partial(
    gf.cross_section.cross_section,
    cladding_layers=cladding_layers_optical_siepic,
    cladding_offsets=cladding_offsets_optical_siepic,
)

strip_bbox = gf.partial(
    strip,
    add_bbox=add_bbox_siepic,
)

metal_routing = partial(
    gf.cross_section.cross_section,
    layer=LAYER.M2_ROUTER,
    width=10.0,
    port_names=gf.cross_section.port_names_electrical,
    port_types=gf.cross_section.port_types_electrical,
    radius=None,
)
heater_metal = partial(
    metal_routing,
    width=4,
    layer=LAYER.M1_HEATER,
)

cross_sections = get_cross_section_factories(sys.modules[__name__])


__all__ = ("add_pins_siepic", "add_pins_bbox_siepic")


if __name__ == "__main__":
    c = gf.c.mzi()
    # c = gf.c.straight(length=1, cross_section=strip)
    # c = gf.c.bend_euler(cross_section=strip)
    # c = gf.c.mzi(delta_length=10, cross_section=strip)
    c.show(show_ports=False)
