"""Customize gdsfactory meep simulations with UBC layer_stack and Path
"""

import meep as mp
import gdsfactory as gf

from gdsfactory.config import logger
from gdsfactory.simulation import plot
from gdsfactory.simulation.gmeep import port_symmetries
import gdsfactory.simulation.gmeep as gm

from ubcpdk.config import PATH
from ubcpdk.tech import LAYER_STACK

sparameters = PATH.sparameters

write_sparameters_meep = gf.partial(
    gm.write_sparameters_meep, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_lr = gf.partial(
    gm.write_sparameters_meep_lr, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_lt = gf.partial(
    gm.write_sparameters_meep_lt, dirpath=sparameters, layer_stack=LAYER_STACK
)


write_sparameters_meep_mpi = gf.partial(
    gm.write_sparameters_meep_mpi, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_mpi_lr = gf.partial(
    gm.write_sparameters_meep_mpi_lr, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_mpi_lt = gf.partial(
    gm.write_sparameters_meep_mpi_lt, dirpath=sparameters, layer_stack=LAYER_STACK
)


write_sparameters_meep_mpi_pool = gf.partial(
    gm.write_sparameters_meep_mpi_pool, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_mpi_pool_lr = gf.partial(
    gm.write_sparameters_meep_mpi_pool_lr, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_mpi_pool_lt = gf.partial(
    gm.write_sparameters_meep_mpi_pool_lt, dirpath=sparameters, layer_stack=LAYER_STACK
)


logger.info(f"Found Meep {mp.__version__!r} installed at {mp.__path__!r}")

__all__ = [
    "write_sparameters_meep",
    "write_sparameters_meep_lr",
    "write_sparameters_meep_lt",
    "write_sparameters_meep_mpi",
    "write_sparameters_meep_mpi_lr",
    "write_sparameters_meep_mpi_lt",
    "write_sparameters_meep_mpi_pool",
    "write_sparameters_meep_mpi_pool_lr",
    "write_sparameters_meep_mpi_pool_lt",
    "plot",
    "port_symmetries",
]
