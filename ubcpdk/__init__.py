"""UBC Siepic Ebeam PDK from edx course."""
import pathlib
import warnings

import gdsfactory as gf
from gdsfactory.config import logger
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from ubcpdk.config import CONFIG, PATH, module
from ubcpdk.tech import LAYER, strip, LAYER_STACK, LAYER_VIEWS
from ubcpdk import components
from ubcpdk import tech
from ubcpdk import data

from ubcpdk.tech import cross_sections

warnings.warn(
    """
The latest versions of ubcpdk work with Python 3.10 and above.
If you are using Python 3.9 or below, please install ubcpdk==1.21.4
However we recommend you to upgrade to Python 3.10 or above so you can use the latest ubcpdk and gdsfactory features.
""",
    stacklevel=2,
)


__version__ = "1.21.4"

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
    base_pdk=gf.get_generic_pdk(),
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
    sparameters_path=PATH.sparameters,
    interconnect_cml_path=PATH.interconnect_cml_path,
    # default_decorator=tech.add_pins_bbox_siepic,
)
PDK.register_cells_yaml(dirpath=pathlib.Path(__file__).parent.absolute())
PDK.activate()


if __name__ == "__main__":
    f = PDK.cells
    print(f.keys())
