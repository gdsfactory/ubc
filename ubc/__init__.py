"""ubc - UBC Siepic Ebeam PDK from edx course"""

import pp
from ubc.add_gc import add_gc, gc_te1550
from ubc.bend_circular import bend_circular
from ubc.layers import LAYER
from ubc.mzi import mzi
from ubc.waveguide import waveguide
from ubc.y_splitter import y_splitter


def mzi_te(**kwargs):
    component = mzi(**kwargs)
    return add_gc(component=component)


def ring_single_te(**kwargs):
    component = pp.c.ring_single(**kwargs)
    return add_gc(component=component)


component_type2factory = dict(
    waveguide=waveguide,
    bend_circular=bend_circular,
    y_splitter=y_splitter,
    mzi=mzi,
    gc_te1550=gc_te1550,
    add_gc=add_gc,
    mzi_te=mzi_te,
    ring_single_te=ring_single_te,
)


__all__ = list(component_type2factory.keys()) + ["LAYER"]
__version__ = "0.0.2"


if __name__ == "__main__":
    # c = mzi_te(delta_length=100)
    c = ring_single_te()
    pp.show(c)
