"""ubc - UBC Siepic Ebeam PDK from edx course"""

from ubc.add_gc import add_gc, gc_te1550
from ubc.bend_circular import bend_circular
from ubc.mzi import mzi
from ubc.waveguide import waveguide
from ubc.y_splitter import y_splitter

component_type2factory = dict(
    waveguide=waveguide,
    bend_circular=bend_circular,
    y_splitter=y_splitter,
    mzi=mzi,
    gc_te1550=gc_te1550,
    add_gc=add_gc,
)


__all__ = list(component_type2factory.keys())
__version__ = "0.0.2"
