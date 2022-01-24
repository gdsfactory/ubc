"""UBC Siepic Ebeam PDK from edx course"""

import gdsfactory as gf
from ubc.config import CONFIG, PATH
from ubc.tech import LAYER, strip

from ubc import da
from ubc import components
from ubc import tech


lys = gf.layers.load_lyp(PATH.lyp)


__all__ = [
    "CONFIG",
    "da",
    "PATH",
    "components",
    "tech",
    "strip",
    "LAYER",
]
__version__ = "0.0.13"
