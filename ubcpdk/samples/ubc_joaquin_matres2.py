"""Includes heaters."""

import gdsfactory as gf

import ubcpdk
import ubcpdk.components as pdk
from ubcpdk.tech import LAYER
from ubcpdk.samples.write_mask import write_mask_gds_with_metadata

size = (605, 410)


def test_mask3():
    """MZI interferometers."""
    mzi = gf.partial(gf.components.mzi, splitter=ubcpdk.components.ebeam_y_1550)
    mzis = [mzi(delta_length=delta_length) for delta_length in [10, 40, 100]]
    mzis_gc = [pdk.add_fiber_array(mzi) for mzi in mzis]

    mzis = [pdk.mzi_heater(delta_length=delta_length) for delta_length in [40]]
    mzis_heater_gc = [
        pdk.add_fiber_array_pads_rf(mzi, optical_routing_type=2) for mzi in mzis
    ]

    c = gf.pack(mzis_gc + mzis_heater_gc, max_size=size)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_3"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask4():
    """Ring resonators."""

    rings = [pdk.ring_single_heater(length_x=length_x) for length_x in [4, 6]]
    rings_gc = [pdk.add_fiber_array_pads_rf(ring) for ring in rings]

    c = gf.pack(rings_gc, max_size=size)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_4"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    # m, tm = test_mask3()
    m, tm = test_mask4()
    m.show()
