"""Technology definitions

- LayerStack
- cross_sections (xs_)
- constants (WIDTH, CLADDING_OFFSET ...)
"""

import pydantic

import gdsfactory as gf
from gdsfactory.tech import LayerStack, LayerLevel
from gdsfactory.types import Layer
import gdsfactory.simulation as sim
import gdsfactory.simulation.lumerical as lumerical
from gdsfactory.add_pins import add_pins_siepic

from ubcpdk.config import PATH

nm = 1e-3

MATERIAL_NAME_TO_LUMERICAL = {
    "si": "Si (Silicon) - Palik",
    "sio2": "SiO2 (Glass) - Palik",
    "sin": "Si3N4 (Silicon Nitride) - Phillip",
}

MATERIAL_NAME_TO_TIDY3D_INDEX = {
    "si": 3.47,
    "sio2": 1.44,
    "sin": 2.0,
}


@pydantic.dataclasses.dataclass(frozen=True)
class LayerMapUbc:
    WG: Layer = (1, 0)
    WG2: Layer = (31, 0)
    DEVREC: Layer = (68, 0)
    LABEL: Layer = (10, 0)
    PORT: Layer = (1, 10)  # PinRec
    PORTE: Layer = (1, 11)  # PinRecM
    FLOORPLAN: Layer = (99, 0)


LAYER = LayerMapUbc()


def get_layer_stack_ubc(thickness: float = 220 * nm) -> LayerStack:
    """Returns generic LayerStack"""
    return LayerStack(
        strip=LayerLevel(
            layer=LAYER.WG,
            thickness=thickness,
            zmin=0.0,
            material="si",
        ),
        strip2=LayerLevel(
            layer=LAYER.WG2,
            thickness=thickness,
            zmin=0.0,
            material="si",
        ),
    )


LAYER_STACK = get_layer_stack_ubc()
layer_set = gf.layers.load_lyp(PATH.lyp)
to_3d = gf.partial(
    gf.export.to_3d,
    layer_set=layer_set,
    layer_stack=LAYER_STACK,
)

write_sparameters_lumerical = gf.partial(
    lumerical.write_sparameters_lumerical,
    layer_stack=LAYER_STACK,
    dirpath=PATH.sparameters,
)

get_sparameters_data_lumerical = gf.partial(
    sim.get_sparameters_data_lumerical,
    layer_stack=LAYER_STACK,
    dirpath=PATH.sparameters,
)


strip = gf.partial(
    gf.cross_section.strip,
    layer=LAYER.WG,
    layer_bbox=LAYER.DEVREC,
    decorator=add_pins_siepic
)
