from typing import Callable, Dict, Tuple

from pp.add_pins import _add_pin_square, _add_pins, _add_pins_labels_and_outline
from pp.component import Component
from ubc.layers import port_type2layer


def add_pins(
    component: Component,
    add_port_marker_function: Callable = _add_pin_square,
    port_type2layer: Dict[str, Tuple[int, int]] = port_type2layer,
):

    for p in component.ports.values():
        layer = port_type2layer[p.port_type]
        add_port_marker_function(
            component=component, port=p, layer=layer, label_layer=layer
        )
    return component


if __name__ == "__main__":
    import pp

    c = pp.c.waveguide()
    cc = add_pins(c)
    cc.show()
