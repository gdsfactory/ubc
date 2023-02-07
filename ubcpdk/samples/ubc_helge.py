import gdsfactory as gf

import ubcpdk
import ubcpdk.components as pdk
from ubcpdk.tech import LAYER
from ubcpdk.samples.write_mask import write_mask_gds_with_metadata

size = (605, 410)
add_gc = ubcpdk.components.add_fiber_array


def test_mask_1():
    """Ring resonators."""

    @gf.cell
    def dbr_filter(n):
        c = gf.Component()

        splitter = pdk.ebeam_bdc_te1550()

        splitter_1 = c << splitter
        splitter_2 = c << splitter

        dbr = pdk.dbg(n=n) if n > 0 else pdk.straight(0)
        dbr_1 = c << dbr
        dbr_2 = c << dbr

        dbr_1.connect("o1", splitter_1["o3"])
        dbr_2.connect("o1", splitter_1["o4"])
        splitter_2.connect("o4", dbr_1["o2"])

        bend_1 = c << pdk.bend(angle=90)
        bend_1.connect("o1", splitter_1["o1"])
        bend_2 = c << pdk.bend(angle=90)
        bend_2.connect("o2", splitter_2["o1"])

        c.add_port("out1", port=bend_2["o1"])
        c.add_port("out2", port=bend_1["o2"])
        c.add_port("in1", port=splitter_1["o2"])

        return c

    rings = [dbr_filter(length) for length in [0, 250, 500, 750, 1000, 1250]]
    rings_gc = [pdk.add_fiber_array(ring, fanout_length=15) for ring in rings]

    c = gf.pack(rings_gc, max_size=size)
    m = c[0]
    m.name = "EBeam_Helge_Simon_1"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask_2():
    e = [add_gc(ubcpdk.components.straight())] * 2
    e += [
        add_gc(
            gf.components.ring_single(
                radius=12,
                gap=gap,
                length_x=coupling_length,
                bend=gf.components.bend_circular,
            )
        )
        for gap in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        for coupling_length in [0, 2]
    ]

    c = gf.pack(e, max_size=size)
    m = c[0]
    m.name = "EBeam_Helge_Simon_2"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    m, _ = test_mask_2()
    m.show()
