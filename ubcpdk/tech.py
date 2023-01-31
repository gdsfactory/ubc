"""Technology definitions.

- LayerStack
- cross_sections (xs_)
- constants (WIDTH, CLADDING_OFFSET ...)

"""
import sys

from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_section_factories
from gdsfactory.technology import LayerStack, LayerLevel
from gdsfactory.types import Layer, LayerSpec, Component, Callable
from gdsfactory.add_pins import add_pin_path, add_pins_siepic

from ubcpdk.config import PATH

nm = 1e-3


class LayerMapUbc(BaseModel):
    WG: Layer = (1, 0)
    WG2: Layer = (31, 0)
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
    remove_layers = [layer_pin, bbox_layer, "TEXT"]
    c = component.remove_layers(layers=remove_layers)
    c.add_padding(default=padding, layers=(bbox_layer,))

    c = add_pins_siepic(
        component=component,
        function=function,
        port_type=port_type,
        layer_pin=layer_pin,
        pin_length=pin_length,
    )
    return c


def get_layer_stack_ubc(thickness: float = 220 * nm) -> LayerStack:
    """Returns UBC LayerStack.

    TODO: Translate xml in lumerical process file.
    """
    return LayerStack(
        layers=dict(
            strip=LayerLevel(
                layer=LAYER.WG,
                thickness=thickness,
                zmin=0.0,
                material="si",
            ),
            strip2=LayerLevel(
                layer=LAYER.WG2,
                thickness=thickness,
                zmin=0.0,
                material="si",
            ),
        )
    )


class Tech(BaseModel):
    name: str = "ubc"
    layer: LayerMapUbc = LAYER

    fiber_array_spacing: float = 250.0
    WG = {"width": 0.5}
    DEVREC = {"width": 0.5}


TECH = Tech()


LAYER_STACK = get_layer_stack_ubc()
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

strip = gf.partial(
    gf.cross_section.cross_section,
    # add_pins=add_pins_siepic_optical_2nm,
    # add_bbox=add_bbox_siepic,
    # cladding_layers=cladding_layers_optical_siepic,
    # cladding_offsets=cladding_offsets_optical_siepic,
    # bbox_layers=cladding_layers_optical_siepic,
    # bbox_offsets=cladding_offsets_optical_siepic,
    # decorator=add_pins_bbox_siepic,
)


cross_sections = get_cross_section_factories(sys.modules[__name__])


__all__ = ("add_pins_siepic", "add_pins_bbox_siepic")


if __name__ == "__main__":
    # c = gf.c.straight(length=1, cross_section=strip)
    # c = gf.c.bend_euler(cross_section=strip)
    c = gf.c.mzi(delta_length=10, cross_section=strip)
    c.show(show_ports=False)
