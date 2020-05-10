"""ubc - UBC Siepic Ebeam PDK from edx course"""

from ubc.bend_circular import bend_circular
from ubc.mzi import mzi
from ubc.waveguide import waveguide
from ubc.y_splitter import y_splitter

component_type2factory = dict(
    waveguide=waveguide, bend_circular=bend_circular, y_splitter=y_splitter, mzi=mzi
)


__all__ = list(component_type2factory.keys())
__version__ = "0.0.1"
