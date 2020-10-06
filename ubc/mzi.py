import pp
from ubc.bend90 import bend90
from ubc.waveguide import waveguide
from ubc.y_splitter import y_splitter

L2 = 0.01
L0 = 0.01


@pp.autoname
def mzi(
    delta_length=100,
    coupler=y_splitter,
    waveguide=waveguide,
    bend90=bend90,
    L2=L2,
    L0=L0,
):
    c = pp.c.mzi(
        DL=delta_length,
        waveguide=waveguide,
        bend90=bend90,
        coupler=coupler,
        L2=L2,
        L0=L0,
    )
    return c


if __name__ == "__main__":
    c = mzi(delta_length=100)
    pp.show(c)
    pp.write_gds(c, "mzi.gds")
