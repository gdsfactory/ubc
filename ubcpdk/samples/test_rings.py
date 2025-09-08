import pathlib
from functools import partial

import gdsfactory as gf

from ubcpdk import PDK, cells

size = (605, 410)
pack = partial(
    gf.pack, max_size=size, add_ports_prefix=False, add_ports_suffix=False, spacing=2
)
add_gc = cells.add_fiber_array

length_x = 0.1


@gf.cell
def EBeam_YourUserName_ring_double10() -> pathlib.Path:
    gaps = [100, 150, 200]
    radiuses = [10]

    rings = [
        cells.ring_double(radius=radius, length_x=length_x, gap=gap * 1e-3)
        for radius in radiuses
        for gap in gaps
    ]
    rings = [add_gc(ring) for ring in rings]
    c = pack(rings)
    if len(c) > 1:
        raise ValueError(f"Failed to pack in 1 component of {size}, got {len(c)}")
    return c[0]


@gf.cell
def EBeam_YourUserName_ring_double30() -> pathlib.Path:
    gaps = [150, 200, 250]
    radiuses = [30]
    rings = [
        cells.ring_double(
            radius=radius,
            length_x=length_x,
            gap=gap * 1e-3,
        )
        for radius in radiuses
        for gap in gaps
    ]

    rings = [add_gc(ring) for ring in rings]

    c = pack(rings)
    if len(c) > 1:
        raise ValueError(f"Failed to pack in 1 component of {size}, got {len(c)}")
    return c[0]


@gf.cell
def EBeam_YourUserName_ring_double3() -> pathlib.Path:
    gaps = [100, 150]
    radiuses = [5]
    rings = [
        cells.ring_double(
            radius=radius,
            length_x=length_x,
            gap=gap * 1e-3,
        )
        for radius in radiuses
        for gap in gaps
    ]

    rings = [add_gc(ring) for ring in rings]

    c = pack(rings)
    if len(c) > 1:
        raise ValueError(f"Failed to pack in 1 component of {size}, got {len(c)}")
    return c[0]


if __name__ == "__main__":
    PDK.activate()
    c = EBeam_YourUserName_ring_double3()
    c.write_gds("extra/EBeam_YourUserName_ring_double3.gds")
    c.show()
