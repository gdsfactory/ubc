import klayout.db as kl
import pp
import ubc
from pp.pack import pack


def change_grid_klayout(
    gdspath, gdspathout=None, precision_in=1e-9, precision_out=1e-9,
):
    """ This script allows you to change the design grid by reading a layout written with different precission (DBU database units)
    and scaling to for example 1nm grid

    Using this script here is a bit of a hack to ensure that this gdsfactory layout works with Calibre.
    Calibre has some issue when there are 2 cells defined in the GDS with the same name
    This script reads a GDS in klayout and writes it again.
    """
    assert precision_in >= precision_out

    gdspathout = gdspathout or gdspath
    gdspath = str(gdspath)
    gdspathout = str(gdspathout)

    layout = kl.Layout()
    layout.read(gdspath)
    layout.top_cell()

    scale = int(precision_in / precision_out)

    layout.dbu = precision_out / 1e-6
    if scale > 1:
        layout.scale_and_snap(layout.top_cell(), 1, scale, 1)
    layout.write(gdspathout)
    return gdspathout


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
    gdspath = pp.write_gds(m, precision=1e-9)
    change_grid_klayout(gdspath)
    pp.show(m)
