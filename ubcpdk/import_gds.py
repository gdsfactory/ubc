from numpy import ndarray
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.add_ports import add_ports_from_siepic_pins

from ubcpdk.tech import LAYER
from ubcpdk.config import PATH


layer = LAYER.WG
port_width = 0.5


def guess_port_orientaton(position: ndarray, name: str, label: str, n: int) -> int:
    """we assume that ports with x<0 are inputs (orientation=180deg)
    and ports with x>0 are outputs
    """
    p = position
    if "gc" in name:
        return 0
    if label == "opt1":
        return 180
    if p[1] > 0 and "crossing" in name:
        return 90
    if p[1] < 0 and "crossing" in name:
        return 270
    if n == 4:
        if label in {"opt1", "opt2"}:
            return 180
        if label in {"opt3", "opt4"}:
            return 0
    if p[0] <= 0:
        return 180
    return 0


def remove_pins(component) -> Component:
    """Remove PINS and"""
    component.remove_layers(layers=(LAYER.DEVREC, LAYER.PORT, LAYER.PORTE))
    component.paths = []
    component._bb_valid = False
    return component


def remove_pins_recursive(component):
    component = remove_pins(component)
    if component.references:
        for ref in component.references:
            rcell = ref.parent
            ref.parent = remove_pins_recursive(rcell)
    return component


def add_ports(component: Component) -> Component:
    """Add ports from labels.
    guessing port orientaton from port location
    """

    c = component
    n = sum(1 for label in c.get_labels() if label.text.startswith("opt"))
    for label in c.get_labels():
        if label.text.startswith("opt"):
            port_name = label.text
            port = gf.Port(
                name=port_name,
                midpoint=label.position,
                width=port_width,
                orientation=guess_port_orientaton(
                    position=label.position,
                    name=c.name,
                    label=label.text,
                    n=n,
                ),
                layer=layer,
            )
            if port_name not in c.ports:
                c.add_port(port)
        return c


add_ports_from_siepic_pins = gf.partial(
    add_ports_from_siepic_pins,
    pin_layer_optical=LAYER.PORT,
    pin_layer_electrical=LAYER.PORTE,
)

import_gds = gf.partial(
    gf.import_gds,
    gdsdir=PATH.gds,
    library="Design kits/ebeam",
    decorator=add_ports_from_siepic_pins,
)

if __name__ == "__main__":
    # gdsname = "ebeam_crossing4.gds"
    gdsname = "ebeam_y_1550.gds"
    c = import_gds(gdsname)
    # print(c.ports)
    c.show(show_ports=False)
