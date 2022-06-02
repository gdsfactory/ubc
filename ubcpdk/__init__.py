"""UBC Siepic Ebeam PDK from edx course"""
import pathlib

import gdsfactory as gf
from gdsfactory.config import logger
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from ubcpdk.config import CONFIG, PATH, module
from ubcpdk.tech import LAYER, strip
from ubcpdk import components
from ubcpdk import tech
from ubcpdk import data

from ubcpdk.tech import cross_sections


gf.asserts.version(">=5.1.2")
lys = gf.layers.load_lyp(PATH.lyp)
__version__ = "1.6.0"

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
    layers=LAYER.dict(),
    base_pdk=gf.pdk.GENERIC,
)
PDK.register_cells_yaml(dirpath=pathlib.Path(__file__).parent.absolute())
PDK.activate()


if __name__ == "__main__":
    f = PDK.cells
    print(f.keys())
