import pathlib
from functools import partial

import gdsfactory as gf

import ubcpdk
import ubcpdk.components as pdk

size = (605, 410)
pack = partial(
    gf.pack, max_size=size, add_ports_prefix=False, add_ports_suffix=False, spacing=2
)
add_gc = ubcpdk.components.add_fiber_array

length_x = 0.1


@gf.cell
def EBeam_JoaquinMatres_1() -> pathlib.Path:
    gaps = [100, 150, 200]
    radiuses = [10]

    rings = [
        pdk.ring_double(
            radius=radius, length_x=length_x, gap=gap * 1e-3, decorator=add_gc
        )
        for radius in radiuses
        for gap in gaps
    ]
    c = pack(rings)
    if len(c) > 1:
        raise ValueError(f"Failed to pack in 1 component of {size}, got {len(c)}")
    return c[0]


@gf.cell
def EBeam_JoaquinMatres_2() -> pathlib.Path:
    gaps = [150, 200, 250]
    radiuses = [30]
    rings = [
        pdk.ring_double(
            radius=radius, length_x=length_x, gap=gap * 1e-3, decorator=add_gc
        )
        for radius in radiuses
        for gap in gaps
    ]

    c = pack(rings)
    if len(c) > 1:
        raise ValueError(f"Failed to pack in 1 component of {size}, got {len(c)}")
    return c[0]


@gf.cell
def EBeam_JoaquinMatres_3() -> pathlib.Path:
    gaps = [100, 150]
    radiuses = [3]
    rings = [
        pdk.ring_double(
            radius=radius, length_x=length_x, gap=gap * 1e-3, decorator=add_gc
        )
        for radius in radiuses
        for gap in gaps
    ]

    c = pack(rings)
    if len(c) > 1:
        raise ValueError(f"Failed to pack in 1 component of {size}, got {len(c)}")
    return c[0]


if __name__ == "__main__":
    c = EBeam_JoaquinMatres_3()
    c.show()
