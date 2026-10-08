import gdsfactory as gf
from ubcpdk import cells


@gf.cell
def sample_fill() -> gf.Component:
    """Sample fill example."""
    c = gf.Component()
    _ = c << cells.die()
    _ = c << cells.add_fiber_array(cells.ring_single())
    fill = gf.c.rectangle(layer="M2_ROUTER")

    c.fill(
        fill_cell=fill,
        fill_layers=[("FLOORPLAN", -10)],
        exclude_layers=[((1, 0), 10), ("M2_ROUTER", 100)],
        x_space=1,
        y_space=1,
    )
    return c
