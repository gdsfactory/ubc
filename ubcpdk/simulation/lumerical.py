"""Customize gdsfactory lumerical simulations with UBC layer_stack and Path
"""
import gdsfactory as gf
from gdsfactory.simulation import plot
import gdsfactory.simulation.lumerical as sim

from ubcpdk.config import PATH
from ubcpdk.tech import LAYER_STACK

sparameters = PATH.sparameters

write_sparameters_lumerical = gf.partial(
    sim.write_sparameters_lumerical, dirpath=sparameters, layer_stack=LAYER_STACK
)


__all__ = [
    "write_sparameters_lumerical",
    "plot",
]
