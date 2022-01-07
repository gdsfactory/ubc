"""Technology definitions

- LayerStack
- cross_sections (xs_)
- constants (WIDTH, CLADDING_OFFSET ...)
- adapted gdsfactory.components to this technology
"""

import pydantic

import gdsfactory as gf
from gdsfactory.tech import LayerStack, LayerLevel
from gdsfactory.types import Layer

from ubc.config import PATH

nm = 1e-3


@pydantic.dataclasses.dataclass(frozen=True)
class LayerMapUbc:
    WG: Layer = (1, 0)
    WG2: Layer = (31, 0)
    DEVREC: Layer = (68, 0)
    LABEL: Layer = (10, 0)
    PORT: Layer = (1, 10)
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


strip = gf.partial(gf.cross_section.strip, layers_cladding=(LAYER.DEVREC,))
