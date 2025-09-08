"""A die with grating couplers and pads."""

import gdsfactory as gf
from gdsfactory.typings import (
    Ints,
    LayerSpec,
    Size,
)

from ubcpdk.tech import LAYER


@gf.cell
def compass(
    size: Size = (4, 2),
    layer: LayerSpec = "MTOP",
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
    layer: LayerSpec = "MTOP",
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
    layer: LayerSpec = "MTOP",
    port_inclusion: float = 0,
    port_orientation: float = 0,
    port_orientations: Ints | None = (180, 90, 0, -90),
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
        port_orientations=port_orientations,
        bbox_layers=(LAYER.PAD_OPEN,),
        bbox_offsets=(-1.8,),
    )


@gf.cell
def die(size: tuple[float, float] = (440, 470), centered: bool = False) -> gf.Component:
    """A die."""
    c = gf.Component()
    _ = c << gf.c.rectangle(
        size=size, layer=LAYER.FLOORPLAN, centered=centered, port_type=None
    )
    return c
