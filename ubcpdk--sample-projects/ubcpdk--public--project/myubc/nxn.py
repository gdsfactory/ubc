"""Write GDS with sample errors."""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec
from ubcpdk import LAYER

layer = LAYER.WG
layer1 = LAYER.WG


@gf.cell
def nxn(
    west: int = 1,
    east: int = 4,
    north: int = 0,
    south: int = 0,
    xsize: float = 8.0,
    ysize: float = 8.0,
    wg_width: float = 0.45,
    layer: LayerSpec = "WG",
    wg_margin: float = 1.0,
) -> Component:
    """Returns nxn component.

    Args:
        west: number of waveguides on the west side.
        east: number of waveguides on the east side.
        north: number of waveguides on the north side.
        south: number of waveguides on the south side.
        xsize: size of the component in x direction.
        ysize: size of the component in y direction.
        wg_width: waveguide width.
        layer: layer of the waveguide.
        wg_margin: margin between waveguides.
    """
    return gf.c.nxn(
        west=west,
        east=east,
        north=north,
        south=south,
        xsize=xsize,
        ysize=ysize,
        wg_width=wg_width,
        layer=layer,
        wg_margin=wg_margin,
    )
