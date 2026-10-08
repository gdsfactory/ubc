"""LVS demo."""

import gdsfactory as gf
from ubcpdk import cells


@gf.cell
def pads_correct(pad=cells.pad, cross_section="metal_routing") -> gf.Component:
    """Returns 2 pads connected with metal wires."""
    c = gf.Component()

    xs = gf.get_cross_section(cross_section)
    layer = gf.get_layer(xs.layer)

    pad = gf.get_component(pad)
    tl = c << pad
    bl = c << pad

    tr = c << pad
    br = c << pad

    tl.name = "tl"
    bl.name = "bl"
    tr.name = "tr"
    br.name = "br"

    tl.move((0, 300))
    br.move((500, 0))
    tr.move((500, 500))

    c.add_label("tl", position=tl.dcenter, layer=layer)
    c.add_label("tr", position=tr.dcenter, layer=layer)
    c.add_label("br", position=br.dcenter, layer=layer)
    c.add_label("bl", position=bl.dcenter, layer=layer)

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    gf.routing.route_bundle_electrical(c, ports1, ports2, cross_section=cross_section)
    return c


@gf.cell
def pads_shorted(pad=cells.pad, cross_section="metal_routing") -> gf.Component:
    """Returns 2 pads connected with metal wires."""
    c = gf.Component()
    pad = gf.get_component(pad)
    xs = gf.get_cross_section(cross_section)
    layer = gf.get_layer(xs.layer)

    tl = c << pad
    bl = c << pad

    tr = c << pad
    br = c << pad

    tl.name = "tl"
    bl.name = "bl"
    tr.name = "tr"
    br.name = "br"

    tl.move((0, 300))
    br.move((500, 0))
    tr.move((500, 500))

    c.add_label("tl", position=tl.dcenter, layer=layer)
    c.add_label("tr", position=tr.dcenter, layer=layer)
    c.add_label("br", position=br.dcenter, layer=layer)
    c.add_label("bl", position=bl.dcenter, layer=layer)

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    gf.routing.route_bundle_electrical(c, ports1, ports2, cross_section=cross_section)

    gf.routing.route_single_electrical(
        c, bl.ports["e2"], tl.ports["e4"], cross_section=cross_section
    )
    return c
