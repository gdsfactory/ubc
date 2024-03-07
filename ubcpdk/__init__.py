"""UBC Siepic Ebeam PDK from edx course."""

from gdsfactory.config import PATH as GPATH
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from ubcpdk import components, data, tech
from ubcpdk.config import CONFIG, PATH
from ubcpdk.tech import LAYER, LAYER_STACK, LAYER_VIEWS, cross_sections

__version__ = "2.4.1"

__all__ = [
    "CONFIG",
    "data",
    "PATH",
    "components",
    "tech",
    "LAYER",
    "cells",
    "cross_sections",
    "PDK",
    "__version__",
]


cells = get_cells(components)
PDK = Pdk(
    name="ubcpdk",
    cells=cells,
    cross_sections=cross_sections,
    layers=dict(LAYER),
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
)

GPATH.sparameters = PATH.sparameters
GPATH.interconnect = PATH.interconnect_cml_path
PDK.activate()


if __name__ == "__main__":
    f = PDK.cells
    for k, _v in f.items():
        print(k)
