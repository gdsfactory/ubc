import gdsfactory as gf
from ubcpdk import cells


@gf.cell
def EBeam_JoaquinMatres_2() -> gf.Component:
    """Spirals for extracting straight waveguide loss."""
    size = (440, 470)
    e = [
        cells.add_fiber_array(component=cells.spiral(n_loops=8, length=length))
        for length in [0, 100, 200]
    ]

    c = gf.Component()
    _ = c << gf.pack(e, max_size=size, spacing=2)[0]
    _ = c << cells.die()
    return c
