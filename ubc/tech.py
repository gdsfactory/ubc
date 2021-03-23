import dataclasses
from typing import Tuple

from pp.tech import Tech
from pp.types import Layer


@dataclasses.dataclass(frozen=True)
class LayerMap:
    WG = (1, 0)
    DEVREC = (68, 0)
    LABEL = (10, 0)
    PORT = (1, 10)
    FLOORPLAN = (99, 0)


LAYER = LayerMap()
port_layer2type = {LAYER.PORT: "optical"}
port_type2layer = {v: k for k, v in port_layer2type.items()}


@dataclasses.dataclass(frozen=True)
class TechSiliconCband(Tech):
    name: str = "silicon_cband"
    wg_width: float = 0.5
    bend_radius: float = 5.0
    cladding_offset: float = 1.0
    layer_wg: Layer = LAYER.WG
    layers_cladding: Tuple[Layer, ...] = (LAYER.DEVREC,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 15.0
    taper_width: float = 2.0  # taper to wider waveguides for lower loss


TECH_SILICON_C = TechSiliconCband()
