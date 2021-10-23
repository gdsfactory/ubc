import pydantic

import gdsfactory as gf
from gdsfactory.tech import LayerStack, LayerLevel
from gdsfactory.types import Layer

nm = 1e-3


@pydantic.dataclasses.dataclass(frozen=True)
class LayerMap:
    WG: Layer = (1, 0)
    WG2: Layer = (31, 0)
    DEVREC = (68, 0)
    LABEL = (10, 0)
    PORT = (1, 10)
    FLOORPLAN = (99, 0)


LAYER = LayerMap()


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


strip = gf.partial(gf.cross_section.strip, layers_cladding=(LAYER.DEVREC,))
