from typing import Tuple, Optional, List
import pydantic.dataclasses as dataclasses

from pp.tech import Waveguide, TECH
from pp.tech import LayerStack, LayerLevel
from pp.types import Layer
from ubc.config import PATH


@dataclasses.dataclass(frozen=True)
class LayerMap:
    WG: Layer = (1, 0)
    DEVREC = (68, 0)
    LABEL = (10, 0)
    PORT = (1, 10)
    FLOORPLAN = (99, 0)


LAYER = LayerMap()
PORT_LAYER_TO_TYPE = {LAYER.PORT: "optical"}
PORT_TYPE_TO_LAYER = {v: k for k, v in PORT_LAYER_TO_TYPE.items()}


@dataclasses.dataclass
class LayerStackUbc(LayerStack):
    WG = LayerLevel((1, 0), thickness_nm=220.0, zmin_nm=0.0, material="si")
    WG2 = LayerLevel((31, 0), thickness_nm=220.0, zmin_nm=0.0, material="si")


LAYER_STACK = LayerStackUbc()


@dataclasses.dataclass
class Strip(Waveguide):
    width: float = 0.5
    width_wide: float = 2.0
    auto_widen: bool = True
    auto_widen_minimum_length: float = 200
    taper_length: float = 10.0
    layer: Layer = LAYER.WG
    radius: float = 10.0
    cladding_offset: float = 3.0
    layer_cladding: Optional[Layer] = LAYER.DEVREC
    layers_cladding: Optional[List[Layer]] = (LAYER.DEVREC,)


@dataclasses.dataclass
class Waveguides:
    strip: Waveguide = Strip()


@dataclasses.dataclass
class SimulationSettings:
    remove_layers: Tuple[Layer, ...] = (LAYER.DEVREC,)
    background_material: str = "sio2"
    port_width: float = 3e-6
    port_height: float = 1.5e-6
    port_extension_um: float = 1.0
    mesh_accuracy: int = 2
    zmargin: float = 1e-6
    ymargin: float = 2e-6
    wavelength_start: float = 1.2e-6
    wavelength_stop: float = 1.6e-6
    wavelength_points: int = 500


WAVEGUIDES = Waveguides()
SIMULATION_SETTINGS = SimulationSettings()


TECH.name = "ubc"
TECH.layer_label = LAYER.LABEL
TECH.layer_stack = LAYER_STACK
TECH.sparameters_path = str(PATH.sparameters)
TECH.waveguide.strip = Strip()
