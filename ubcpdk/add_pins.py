from typing import Callable, Optional

from gdsfactory.add_pins import add_pin_path
from gdsfactory.component import Component
from gdsfactory.types import Layer
from ubcpdk.tech import LAYER


def add_pins(
    component: Component,
    function: Callable = add_pin_path,
    layer_port: Layer = LAYER.WG,
    layer_pin: Layer = LAYER.PORT,
    bbox_layer: Optional[Layer] = LAYER.DEVREC,
) -> Component:
    """Add pins and device recognition layer.

    Args:
        component: to add pins
        function:
        layer_port:
        layer_pin:
        bbox_layer: bounding box layer
    """
    if bbox_layer:
        component.add_padding(default=0, layers=(bbox_layer,))

    for p in component.get_ports_list(layer=layer_port):
        function(component=component, port=p, layer=layer_pin, layer_label=layer_pin)

    return component


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.crossing()
    c.unlock()
    add_pins(c)
    c.lock()
    c.show(show_ports=False)
