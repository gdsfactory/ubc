import pp
from ubc.bend90 import bend90
from ubc.waveguide import waveguide
from ubc.y_splitter import y_splitter


@pp.autoname
def mzi(
    delta_length=100,
    coupler_factory=y_splitter,
    straight_factory=waveguide,
    bend90_factory=bend90,
):
    c = pp.c.mzi(
        DL=delta_length,
        straight_factory=straight_factory,
        bend90_factory=bend90_factory,
        coupler_factory=coupler_factory,
    )
    return c


if __name__ == "__main__":
    c = mzi(delta_length=100)
    pp.show(c)
    pp.write_gds(c, "mzi.gds")
