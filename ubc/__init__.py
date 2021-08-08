"""UBC Siepic Ebeam PDK from edx course"""

import gdsfactory as gf
from ubc.config import CONFIG, PATH
from ubc.write_sparameters import write_sparameters

from ubc import da
from ubc import components


lys = gf.layers.load_lyp(PATH.lyp)


__all__ = ["CONFIG", "da", "PATH", "components", "LIBRARY", "write_sparameters"]
__version__ = "0.0.3"
