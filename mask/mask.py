import pp
import ubc
from pp.pack import pack


def add_floorplan(c, size=(605, 410), layer=ubc.LAYER.FLOORPLAN):
    c << pp.c.rectangle(size=size, layer=layer)


if __name__ == "__main__":
    e = [ubc.mzi_te(delta_length=dl) for dl in [10, 100]]
    e += [
        ubc.ring_single_te(bend_radius=10, gap=gap, length_x=coupling_length)
        for gap in [0.2]
        for coupling_length in [1e-3, 1, 2]
    ]
    c = pack(e)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_1"
    add_floorplan(m)
    pp.write_gds(m)
    pp.show(m)
