"""UBC Siepic Ebeam PDK from edx course"""

import gdsfactory as gf
from ubcpdk.config import CONFIG, PATH
from ubcpdk.tech import LAYER, strip
from ubcpdk import components
from ubcpdk import tech
from ubcpdk import data


lys = gf.layers.load_lyp(PATH.lyp)


__all__ = [
    "CONFIG",
    "data",
    "PATH",
    "components",
    "tech",
    "strip",
    "LAYER",
]
__version__ = "1.0.1"
