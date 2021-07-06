from .add_grating_coupler import add_grating_coupler
from .mzi import mzi
from .mzi_spectrum import mzi_spectrum
import gdslib

circuit_factory = dict(mzi=mzi)
component_factory = dict()

circuit_names = list(circuit_factory.keys())
component_names = list(component_factory.keys())


__all__ = ["add_grating_coupler", "mzi", "mzi_spectrum"]


if gdslib.__version__ < "0.3.0":
    raise ValueError(
        f"gdslib version = {gdslib.__version__}"
        " you need to upgrade gdslib `pip install gdslib`"
    )
