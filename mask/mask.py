import pp
import ubc
from pp.pack import pack


def add_floorplan(c, size=(605, 410), layer=ubc.LAYER.FLOORPLAN):
    c << pp.c.rectangle(size=size, layer=layer)


if __name__ == "__main__":
    mzis = [ubc.mzi_te(delta_length=dl) for dl in [10, 100]]
    c = pack(mzis)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_1"
    add_floorplan(m)
    pp.write_gds(m)
    pp.show(m)
