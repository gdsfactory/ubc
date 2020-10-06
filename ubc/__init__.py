"""ubc - UBC Siepic Ebeam PDK from edx course"""

import pp
import ubc.da as da
from ubc.add_gc import (
    add_gc,
    gc_te1310,
    gc_te1550,
    gc_te1550_broadband,
    gc_tm1550,
    taper_factory,
)
from ubc.bend90 import bend90
from ubc.config import conf, path
from ubc.crossing import crossing_te, crossing_te_ring
from ubc.dbr import dbr_te
from ubc.dcate import dcate
from ubc.dcbte import dcbte
from ubc.layers import LAYER
from ubc.mzi import mzi
from ubc.ring import ring
from ubc.waveguide import waveguide
from ubc.y_adiabatic import y_adiabatic
from ubc.y_splitter import y_splitter


def mzi_te(**kwargs):
    component = mzi(**kwargs)
    return add_gc(component=component)


def spiral_te(**kwargs):
    c = pp.c.spiral_external_io(**kwargs)
    length = c.settings["total_length"]
    return add_gc(component=c, component_name=f"spiral_te_{int(length)}")


def ring_single_te(**kwargs):
    component = pp.c.ring_single(**kwargs)
    return add_gc(component=component)


def cavity_te(**kwargs):
    component = pp.c.cavity(**kwargs)
    return add_gc(component=component)


_component_functions = [
    bend90,
    crossing_te,
    gc_te1550,
    ring,
    waveguide,
    y_splitter,
]  # for the klayout library


component_type2factory = dict(
    bend90=bend90,
    crossing_te=crossing_te,
    crossing_te_ring=crossing_te_ring,
    dbr_te=dbr_te,
    dcate=dcate,
    dcbte=dcbte,
    gc_te1550=gc_te1550,
    gc_te1550_broadband=gc_te1550_broadband,
    gc_te1310=gc_te1310,
    gc_tm1550=gc_tm1550,
    mzi=mzi,
    mzi_te=mzi_te,
    ring=ring,
    ring_single_te=ring_single_te,
    taper_factory=taper_factory,
    waveguide=waveguide,
    y_adiabatic=y_adiabatic,
    y_splitter=y_splitter,
)


__all__ = list(component_type2factory.keys()) + [
    "LAYER",
    "conf",
    "path",
    "da",
]
__version__ = "0.0.2"
_components = list(component_type2factory.keys())


if __name__ == "__main__":
    # c = mzi_te(delta_length=100)
    # c = spiral_te(
    #     N=10, x_inner_length_cutback=1, bend_radius=10, y_straight_inner_top=600
    # )

    # N = 15
    # bend_radius = 20
    # c = spiral_te(
    #     N=N,
    #     x_inner_length_cutback=0,
    #     bend_radius=bend_radius,
    #     y_straight_inner_top=0,
    #     x_inner_offset=100,
    # )
    # c = cavity_te(mirror=pp.c.dbr())
    # print(c.settings['component'])
    c = ring_single_te()
    pp.show(c)
