from numpy import ndarray
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.add_ports import add_ports_from_siepic_pins

from ubcpdk.tech import LAYER
from ubcpdk.config import PATH


layer = LAYER.WG
port_width = 0.5


def guess_port_orientaton(position: ndarray, name: str, label: str, n: int) -> int:
    """Assumes ports with x<0 have orientation=180 and ports with x>0  orientation=0."""
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
    return 180 if p[0] <= 0 else 0


def remove_pins(component) -> Component:
    """Remove PINS."""
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

    guess port orientaton from port location.
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
    port_layer_optical=LAYER.WG,
)


def import_gds(gdspath, **kwargs):
    return gf.import_gds(
        gdspath,
        gdsdir=PATH.gds,
        library="Design kits/ebeam",
        model=gdspath.split(".")[0],
        decorator=add_ports_from_siepic_pins,
        **kwargs
    )


def import_gc(gdspath, **kwargs):
    c = import_gds(gdspath, **kwargs)
    return c.mirror().flatten()


if __name__ == "__main__":
    from gdsfactory.write_cells import get_import_gds_script

    script = get_import_gds_script(dirpath=PATH.gds, module="ubcpdk.components")
    print(script)

    # gdsname = "ebeam_crossing4.gds"
    # gdsname = "ebeam_y_1550.gds"
    # c = import_gds(gdsname)
    # print(c.ports)
    # c.show(show_ports=False)
