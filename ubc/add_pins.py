from typing import Callable, Dict, Tuple

from pp.add_pins import add_pin_square
from pp.component import Component
from ubc.tech import PORT_TYPE_TO_LAYER


def add_pins(
    component: Component,
    function: Callable = add_pin_square,
    port_type2layer: Dict[str, Tuple[int, int]] = PORT_TYPE_TO_LAYER,
) -> None:

    for p in component.ports.values():
        layer = port_type2layer[p.port_type]
        function(component=component, port=p, layer=layer, label_layer=layer)


if __name__ == "__main__":
    import pp

    c = pp.c.waveguide()
    cc = add_pins(c)
    cc.show()
