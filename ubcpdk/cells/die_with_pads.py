"""A die with grating couplers and pads."""

import gdsfactory as gf
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Ints,
    LayerSpec,
    Size,
)

from ubcpdk.tech import LAYER


@gf.cell
def compass(
    size: Size = (4, 2),
    layer: LayerSpec = "PAD",
    port_type: str | None = None,
    port_inclusion: float = 0.0,
    port_orientations: Ints | None = (180, 90, 0, -90),
    auto_rename_ports: bool = True,
) -> gf.Component:
    """Rectangle with ports on each edge (north, south, east, and west).

    Args:
        size: rectangle size.
        layer: tuple (int, int).
        port_type: optical, electrical.
        port_inclusion: from edge.
        port_orientations: list of port_orientations to add. None does not add ports.
        auto_rename_ports: auto rename ports.
    """
    return gf.c.compass(
        size=size,
        layer=layer,
        port_type=port_type,
        port_inclusion=port_inclusion,
        port_orientations=port_orientations,
        auto_rename_ports=auto_rename_ports,
    )


@gf.cell
def rectangle(
    size: Size = (4, 2),
    layer: LayerSpec = "PAD",
    centered: bool = False,
    port_type: str | None = None,
    port_orientations: Ints | None = (180, 90, 0, -90),
) -> gf.Component:
    """Returns a rectangle.

    Args:
        size: (tuple) Width and height of rectangle.
        layer: Specific layer to put polygon geometry on.
        centered: True sets center to (0, 0), False sets south-west to (0, 0).
        port_type: optical, electrical.
        port_orientations: list of port_orientations to add. None adds no ports.
    """
    return gf.c.rectangle(
        size=size,
        layer=layer,
        centered=centered,
        port_type=port_type,
        port_orientations=port_orientations,
    )


@gf.cell
def pad(
    size: tuple[float, float] = (90.0, 90.0),
    layer: LayerSpec = "PAD",
    port_inclusion: float = 0,
    port_orientation: float = 0,
) -> gf.Component:
    """Returns rectangular pad with ports.

    Args:
        size: x, y size.
        layer: pad layer.
        bbox_layers: list of layers.
        bbox_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        port_inclusion: from edge.
        port_orientation: in degrees.
    """
    return gf.components.pad(
        size=size,
        layer=layer,
        port_inclusion=port_inclusion,
        port_orientation=port_orientation,
    )


@gf.cell
def die(size: tuple[float, float] = (16000.0, 1 * 3000.0)) -> gf.Component:
    """A die with grating couplers and pads.

    Args:
        size: the size of the die, in um.
        ngratings: the number of grating couplers.
        npads: the number of pads.
        grating_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        grating_coupler: the grating coupler component. None skips the grating couplers.
        cross_section: the cross section.
        pad: the pad component.
        layer_floorplan: the layer of the floorplan.
        edge_to_pad_distance: the distance from the edge to the pads, in um.
        edge_to_grating_distance: the distance from the edge to the grating couplers, in um.
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
    """
    c = gf.Component()
    _ = c << gf.c.rectangle(
        size=size, layer=LAYER.FLOORPLAN, centered=True, port_type=None
    )
    return c


@gf.cell
def die_with_pads(
    size: tuple[float, float] = (11470.0, 4900.0),
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec | None = "grating_coupler_rectangular",
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    layer_floorplan: LayerSpec = "FLOORPLAN",
    edge_to_pad_distance: float = 150.0,
    edge_to_grating_distance: float = 150.0,
    with_loopback: bool = True,
) -> gf.Component:
    """A die with grating couplers and pads.

    Args:
        size: the size of the die, in um.
        ngratings: the number of grating couplers.
        npads: the number of pads.
        grating_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        grating_coupler: the grating coupler component. None skips the grating couplers.
        cross_section: the cross section.
        pad: the pad component.
        layer_floorplan: the layer of the floorplan.
        edge_to_pad_distance: the distance from the edge to the pads, in um.
        edge_to_grating_distance: the distance from the edge to the grating couplers, in um.
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
    """
    c = gf.Component()
    fp = c << gf.c.rectangle(
        size=size, layer=layer_floorplan, centered=True, port_type=None
    )
    xs, ys = size
    x0 = xs / 2 + edge_to_grating_distance
    if grating_coupler:
        gca = gf.c.grating_coupler_array(
            n=ngratings,
            pitch=grating_pitch,
            with_loopback=with_loopback,
            grating_coupler=grating_coupler,
            cross_section=cross_section,
        )
        left = c << gca
        left.rotate(-90)
        left.xmin = -xs / 2 + edge_to_grating_distance
        left.y = fp.y
        c.add_ports(left.ports, prefix="W")
        right = c << gca
        right.rotate(+90)
        right.xmax = xs / 2 - edge_to_grating_distance
        right.y = fp.y
        c.add_ports(right.ports, prefix="E")
    pad = gf.get_component(pad)
    x0 = -npads * pad_pitch / 2 + edge_to_pad_distance
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymax = ys / 2 - edge_to_pad_distance
        c.add_port(name=f"N{i}", port=pad_ref.ports["e4"])
    x0 = -npads * pad_pitch / 2 + edge_to_pad_distance
    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymin = -ys / 2 + edge_to_pad_distance
        c.add_port(name=f"S{i}", port=pad_ref.ports["e2"])
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    from ubcpdk import PDK

    PDK.activate()
    c = die()
    c.pprint_ports()
    c.show()
