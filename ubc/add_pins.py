from typing import Callable

from gdsfactory.add_pins import add_pin_square
from gdsfactory.component import Component
from gdsfactory.types import Layer
from ubc.tech import LAYER


def add_pins(
    component: Component,
    function: Callable = add_pin_square,
    layer_port: Layer = LAYER.WG,
    layer_pin: Layer = LAYER.PORT,
    layer_label: Layer = LAYER.LABEL,
) -> None:

    for p in component.get_ports_list(layer=layer_port):
        function(component=component, port=p, layer=layer_pin, layer_label=layer_label)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.c.straight()
    add_pins(c)
    c.show(show_ports=False)
