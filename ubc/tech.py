from dataclasses import dataclass
from typing import Optional, Tuple

from pp.tech import Tech
from pp.types import Layer
from ubc.layers import LAYER


@dataclass(frozen=True)
class TechSiliconCband(Tech):
    name: str = "silicon_cband"
    wg_width: float = 0.5
    bend_radius: float = 10.0
    cladding_offset: float = 1.0
    layer_wg: Layer = LAYER.WG
    layers_cladding: Tuple[Layer, ...] = (LAYER.DEVREC,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 15.0
    taper_width: float = 2.0  # taper to wider waveguides for lower loss


TECH_SILICON_C = TechSiliconCband()
