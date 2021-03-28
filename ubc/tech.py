import dataclasses
import pathlib
from typing import Tuple

from pp.tech import Tech
from pp.types import Layer
from pp.layers import LayerStack, LayerLevel
from ubc.config import PATH


@dataclasses.dataclass(frozen=True)
class LayerMap:
    WG = (1, 0)
    DEVREC = (68, 0)
    LABEL = (10, 0)
    PORT = (1, 10)
    FLOORPLAN = (99, 0)


LAYER = LayerMap()
PORT_LAYER_TO_TYPE = {LAYER.PORT: "optical"}
PORT_TYPE_TO_LAYER = {v: k for k, v in PORT_LAYER_TO_TYPE.items()}


@dataclasses.dataclass
class LayerStackUbc(LayerStack):
    WG = LayerLevel((1, 0), thickness_nm=220.0, z_nm=0.0, material="si")
    WG2 = LayerLevel((31, 0), thickness_nm=220.0, z_nm=0.0, material="si")


LAYER_STACK = LayerStackUbc()


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
    layer_stack: LayerStack = LAYER_STACK
    sparameters_path: pathlib.Path = PATH.sparameters


TECH_SILICON_C = TechSiliconCband()
