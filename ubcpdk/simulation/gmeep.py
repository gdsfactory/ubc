"""Customize gdsfactory meep simulations with layer_stack and Path
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
write_sparameters_meep_1x1 = gf.partial(
    gm.write_sparameters_meep_1x1, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_1x1_bend90 = gf.partial(
    gm.write_sparameters_meep_1x1_bend90, dirpath=sparameters, layer_stack=LAYER_STACK
)


write_sparameters_meep_mpi = gf.partial(
    gm.write_sparameters_meep_mpi, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_mpi_1x1 = gf.partial(
    gm.write_sparameters_meep_mpi_1x1, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_mpi_1x1_bend90 = gf.partial(
    gm.write_sparameters_meep_mpi_1x1_bend90,
    dirpath=sparameters,
    layer_stack=LAYER_STACK,
)


write_sparameters_meep_batch = gf.partial(
    gm.write_sparameters_meep_batch, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_batch_1x1 = gf.partial(
    gm.write_sparameters_meep_batch_1x1, dirpath=sparameters, layer_stack=LAYER_STACK
)
write_sparameters_meep_batch_1x1_bend90 = gf.partial(
    gm.write_sparameters_meep_batch_1x1_bend90,
    dirpath=sparameters,
    layer_stack=LAYER_STACK,
)


logger.info(f"Found Meep {mp.__version__!r} installed at {mp.__path__!r}")

__all__ = [
    "write_sparameters_meep",
    "write_sparameters_meep_1x1",
    "write_sparameters_meep_1x1_bend90",
    "write_sparameters_meep_mpi",
    "write_sparameters_meep_mpi_1x1",
    "write_sparameters_meep_mpi_1x1_bend90",
    "write_sparameters_meep_batch",
    "write_sparameters_meep_batch_1x1",
    "write_sparameters_meep_batch_1x1_bend90",
    "plot",
    "port_symmetries",
]
