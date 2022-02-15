"""UBC Siepic Ebeam PDK from edx course"""
import gdsfactory as gf
from gdsfactory.config import logger
from ubcpdk.config import CONFIG, PATH, module
from ubcpdk.tech import LAYER, strip
from ubcpdk import components
from ubcpdk import tech
from ubcpdk import data


gf.asserts.version(">=4.1.0")
lys = gf.layers.load_lyp(PATH.lyp)
__version__ = "1.2.1"

__all__ = [
    "CONFIG",
    "data",
    "PATH",
    "components",
    "tech",
    "strip",
    "LAYER",
    "__version__",
]


logger.info(f"Found UBCpdk {__version__!r} installed at {module!r}")
