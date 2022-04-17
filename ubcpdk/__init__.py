"""UBC Siepic Ebeam PDK from edx course"""
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


gf.asserts.version(">=4.7.3")
lys = gf.layers.load_lyp(PATH.lyp)
__version__ = "1.5.0"

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

PDK = Pdk(name="ubcpdk", cells=cells, cross_sections=cross_sections)
PDK.activate()


if __name__ == "__main__":
    f = cells
    print(f.keys())
