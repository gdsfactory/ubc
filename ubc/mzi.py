import pp
from ubc.bend90 import bend90
from ubc.waveguide import waveguide
from ubc.y_splitter import y_splitter


@pp.cell
def mzi(
    delta_length=100,
    splitter=y_splitter,
    waveguide=waveguide,
    bend90=bend90,
    length_y: float = 4.0,
    length_x: float = 0.1,
):
    c = pp.c.mzi(
        delta_length=delta_length,
        waveguide=waveguide,
        bend90=bend90,
        splitter=splitter,
        length_x=length_x,
        length_y=length_y,
    )
    return c


if __name__ == "__main__":
    c = mzi(delta_length=100)
    pp.show(c)
