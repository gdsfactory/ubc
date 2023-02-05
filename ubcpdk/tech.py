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
from gdsfactory.types import Layer, LayerSpec, Component, Callable
from gdsfactory.add_pins import add_pin_path, add_pins_siepic, add_bbox_siepic

from ubcpdk.config import PATH

nm = 1e-3


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
    TEXT: Layer = (66, 0)
    LABEL_INSTANCE: Layer = (66, 0)
    SHOW_PORTS: Layer = (1, 13)
    PADDING: Layer = (67, 0)
    SLAB150: Layer = (2, 0)
    WAFER: Layer = (99999, 0)

    class Config:
        frozen = True
        extra = "forbid"


LAYER = LayerMapUbc()


def add_pins_bbox_siepic(
    component: Component,
    function: Callable = add_pin_path,
    port_type: str = "optical",
    layer_pin: LayerSpec = "PORT",
    pin_length: float = 2 * nm,
    bbox_layer: LayerSpec = "DEVREC",
    padding: float = 0,
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
    """
    remove_layers = (layer_pin, bbox_layer, "TEXT")
    c = component.remove_layers(layers=remove_layers)
    # c = component
    c.add_padding(default=padding, layers=(bbox_layer,))

    c = add_pins_siepic(
        component=component,
        function=function,
        port_type=port_type,
        layer_pin=layer_pin,
        pin_length=pin_length,
    )
    return c


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
    add_bbox=add_bbox_siepic,
    # cladding_layers=cladding_layers_optical_siepic,
    # cladding_offsets=cladding_offsets_optical_siepic,
    # bbox_layers=cladding_layers_optical_siepic,
    # bbox_offsets=cladding_offsets_optical_siepic,
    # decorator=add_pins_bbox_siepic,
)
strip_heater_metal = partial(
    gf.cross_section.strip_heater_metal,
    layer="WG",
    heater_width=2.5,
    layer_heater=LAYER.M1_HEATER,
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
    # c = gf.c.straight(length=1, cross_section=strip)
    # c = gf.c.bend_euler(cross_section=strip)
    c = gf.c.mzi(delta_length=10, cross_section=strip)
    c.show(show_ports=False)
