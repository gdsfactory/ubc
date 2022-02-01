from numpy import ndarray
import gdsfactory as gf
from gdsfactory.component import Component

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
    if n == 4 and label in ["opt1", "opt2"]:
        return 180
    if n == 4 and label in ["opt3", "opt4"]:
        return 0
    if p[0] <= 0:
        return 180
    return 0


def add_ports(component: Component) -> Component:
    """Add ports from labels."""

    c = component
    n = 0
    for label in c.get_labels():
        if label.text.startswith("opt"):
            n += 1

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


add_ports_renamed = gf.compose(gf.port.auto_rename_ports, add_ports)

import_gds = gf.partial(gf.import_gds, gdsdir=PATH.gds, decorator=add_ports_renamed)


if __name__ == "__main__":
    gdsname = "ebeam_y_1550.gds"
    c = import_gds(gdsname)
    print(c.ports)
    c.show()
