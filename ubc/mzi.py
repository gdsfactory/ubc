import pp
from ubc.bend_circular import bend_circular
from ubc.waveguide import waveguide
from ubc.y_splitter import y_splitter


@pp.autoname
def mzi(
    delta_length=100,
    coupler_factory=y_splitter,
    straight_factory=waveguide,
    bend90_factory=bend_circular,
):
    c = pp.c.mzi(
        L1=delta_length,
        straight_factory=straight_factory,
        bend90_factory=bend90_factory,
        coupler_factory=coupler_factory,
    )
    return c


if __name__ == "__main__":
    c = mzi(delta_length=100)
    pp.show(c)
    pp.write_gds(c, "mzi.gds")
