"""UBC Siepic Ebeam PDK from edx course."""

from typing import cast

from gdsfactory.config import PATH as GPATH
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk
from gdsfactory.typings import (
    ConnectivitySpec,
)

from ubcpdk import cells, data, tech
from ubcpdk.config import CONFIG, PATH
from ubcpdk.models import get_models
from ubcpdk.tech import (
    LAYER,
    LAYER_STACK,
    LAYER_VIEWS,
    cross_sections,
    routing_strategies,
)

components = cells

__version__ = "3.2.0"
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

connectivity = cast(list[ConnectivitySpec], [("M1_HEATER", "M1_HEATER", "M2_ROUTER")])

_cells = get_cells(cells)
PDK = Pdk(
    name="ubcpdk",
    cells=_cells,
    cross_sections=cross_sections,
    models=get_models(),
    layers=LAYER,
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
    connectivity=connectivity,
    routing_strategies=routing_strategies,
)

GPATH.sparameters = PATH.sparameters
GPATH.interconnect = PATH.interconnect_cml_path
