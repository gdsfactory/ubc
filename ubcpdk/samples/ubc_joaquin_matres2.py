"""Includes heaters."""

import gdsfactory as gf

import ubcpdk
from ubcpdk.tech import LAYER
from ubcpdk.samples.write_mask import write_mask_gds_with_metadata

size = (605, 410)
add_gc = ubcpdk.components.add_fiber_array


def test_mask3():
    """Add MZI interferometers."""
    mzi = gf.partial(gf.components.mzi, splitter=ubcpdk.components.ebeam_y_1550)
    mzis = [mzi(delta_length=delta_length) for delta_length in [10, 100]]
    mzis_with_gc = [add_gc(mzi) for mzi in mzis]

    c = gf.pack(mzis_with_gc)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_3"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    m, tm = test_mask3()
    m.show()
