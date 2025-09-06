"""UBC Siepic Ebeam PDK from edx course."""

from gdsfactory.config import PATH as GPATH
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from ubcpdk import cells, data, tech
from ubcpdk.config import CONFIG, PATH
from ubcpdk.tech import LAYER, LAYER_STACK, LAYER_VIEWS, cross_sections

try:
    from gplugins.sax.models import get_models

    from ubcpdk import models

    models = get_models(models)
except ImportError:
    print("gplugins[sax] not installed, no simulation models available.")
    models = {}

components = cells

__version__ = "2.7.0"
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


_cells = get_cells(cells)
PDK = Pdk(
    name="ubcpdk",
    cells=_cells,
    cross_sections=cross_sections,
    models=models,
    layers=LAYER,
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
)

GPATH.sparameters = PATH.sparameters
GPATH.interconnect = PATH.interconnect_cml_path
