"""ubc - UBC Siepic Ebeam PDK from edx course"""

from bend_circular import bend_circular
from mzi import mzi
from waveguide import waveguide
from y_splitter import y_splitter

component_type2factory = dict(
    waveguide=waveguide, bend_circular=bend_circular, y_splitter=y_splitter, mzi=mzi
)


__all__ = list(component_type2factory.keys())
__version__ = "0.0.1"
