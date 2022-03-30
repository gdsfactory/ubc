"""UBC Siepic Ebeam PDK from edx course"""
import gdsfactory as gf
from gdsfactory.config import logger
from ubcpdk.config import CONFIG, PATH, module
from ubcpdk.tech import LAYER, strip
from ubcpdk import components
from ubcpdk import tech
from ubcpdk import data

from ubcpdk.components import component_factory
from ubcpdk.tech import cross_section_factory


gf.asserts.version(">=4.2.1")
lys = gf.layers.load_lyp(PATH.lyp)
__version__ = "1.3.8"

__all__ = [
    "CONFIG",
    "data",
    "PATH",
    "components",
    "tech",
    "strip",
    "LAYER",
    "__version__",
    "component_factory",
    "cross_section_factory",
]


logger.info(f"Found UBCpdk {__version__!r} installed at {module!r}")
