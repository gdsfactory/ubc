"""UBC Siepic Ebeam PDK from edx course."""
import pathlib

from gdsfactory.config import PATH as GPATH
from gdsfactory.config import logger
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from ubcpdk import components, data, tech
from ubcpdk.config import CONFIG, PATH, module
from ubcpdk.tech import LAYER, LAYER_STACK, LAYER_VIEWS, cross_sections, strip

__version__ = "2.1.1"

__all__ = [
    "CONFIG",
    "data",
    "PATH",
    "components",
    "tech",
    "strip",
    "LAYER",
    "__version__",
    "cells",
    "cross_sections",
    "PDK",
]


logger.info(f"Found UBCpdk {__version__!r} installed at {module!r}")
cells = get_cells(components)
PDK = Pdk(
    name="ubcpdk",
    cells=cells,
    cross_sections=cross_sections,
    layers=dict(LAYER),
    base_pdk=get_generic_pdk(),
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
)

GPATH.sparameters = PATH.sparameters
GPATH.interconnect = PATH.interconnect_cml_path
PDK.register_cells_yaml(dirpath=pathlib.Path(__file__).parent.absolute())
PDK.activate()


if __name__ == "__main__":
    f = PDK.cells
    print(f.keys())
