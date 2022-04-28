from numpy import ndarray
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.types import Layer

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


def add_ports_from_siepic_pins(
    component: Component,
    optical_pin_layer: Layer = LAYER.PORT,
    electrical_pin_layer: Layer = LAYER.PORTE,
) -> Component:
    """Add ports from SiEPIC-type cells.
    Looks for label, path pairs

    Args:
        component: component
        optical_pin_layer: layer for optical pins
        electrical_pin_layer: layer for electrical pins
    """
    pin_layers = {"optical": optical_pin_layer, "electrical": electrical_pin_layer}
    from numpy import arctan2, degrees, isclose

    # TODO: Add opt-in ports for Lumerical Interconnect simulations
    #   ref: https://github.com/SiEPIC/SiEPIC-Tools/wiki/SiEPIC-Tools-Menu-descriptions#connectivity-layout-check
    # Counters are being used for importing labels, this is useful if the labels follow
    # a different naming convention and names are shared between pins (i.e. >1 pin named 'anode')
    c = component
    labels = c.get_labels()

    for path in c.paths:
        p1, p2 = path.points

        # Find the midpoint of the path
        midpoint = (p1 + p2) / 2

        # Find the label closest to the pin
        label = None
        for i, l in enumerate(labels):
            if all(isclose(l.position, midpoint)):
                label = l
                labels.pop(i)
        if label is None:
            print(f"Warning: label not found for path: ({p1}, {p2})")
            continue
        if optical_pin_layer[0] in path.layers:
            port_type = "optical"
        elif electrical_pin_layer[0] in pin_layers:
            port_type = "electrical"
        else:
            continue

        port_name = str(label.text)

        # If the port name is already used, add a number to it
        i = 1
        while port_name in c.ports:
            port_name += f"_{i}"

        angle = round(degrees(arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 360)
        port = Port(
            name=port_name,
            midpoint=midpoint,
            width=path.widths[0][0],
            orientation=angle,
            layer=pin_layers[port_type],
            port_type=port_type,
        )

        c.add_port(port)
    return c


def add_siepic_labels_and_simulation_info(
    component: Component,
    model: str = None,
    library: str = "Design kits/ebeam",
    label_layer: Layer = LAYER.DEVREC,
) -> Component:
    """

    Args:
        component: component
        model: name of component for SiEPIC label (defaults to component name)
        library: Lumerical Interconnect library for SiEPIC label
        label_layer: layer for writing SiEPIC labels
    """
    c = component

    model = model or c.name
    c.add_label(
        text=f"Component={model}",
        position=c.center + (0, c.size_info.height / 6),
        layer=label_layer,
    )
    c.add_label(
        text=f"Lumerical_INTERCONNECT_library={library}",
        position=c.center - (0, c.size_info.height / 6),
        layer=label_layer,
    )
    return c


import_gds = gf.partial(
    gf.import_gds,
    gdsdir=PATH.gds,
    library="Design kits/ebeam",
    decorator=add_ports_from_siepic_pins,
)
#
# gratings have a 2nm square that is sticking out 1nm
# add_pins_gratings = gf.partial(add_pins_bbox_siepic, padding=-1e-3)
#
# add_ports_renamed = gf.compose(
#     add_pins_bbox_siepic, gf.port.auto_rename_ports, remove_pins, add_ports_from_siepic_pins
# )
# add_ports_renamed_gratings = gf.compose(
#     add_pins_gratings, gf.port.auto_rename_ports, remove_pins, add_ports_from_siepic_pins
# )
#
# # add_ports_siepic = gf.compose(
# #     add_pins_bbox_siepic,
# #     remove_pins,
# #     add_ports_from_siepic_pins,
# # )
# #
# add_ports_siepic_gratings = gf.compose(
#     add_pins_gratings,
#     remove_pins,
#     add_ports_from_siepic_pins,
# )
#
# import_gds_siepic_pins = gf.partial(import_gds, gdsdir=PATH.gds)
#
# import_gds_siepic_pins_gratings = gf.partial(
#     import_gds_siepic_pins,
#     decorator=add_ports_siepic_gratings
# )

if __name__ == "__main__":
    # gdsname = "ebeam_crossing4.gds"
    gdsname = "ebeam_y_1550.gds"
    c = import_gds(gdsname)
    # print(c.ports)
    c.show(show_ports=False)
