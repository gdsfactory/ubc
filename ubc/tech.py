from typing import Tuple
import pydantic

import gdsfactory as gf
from gdsfactory.tech import LayerStack, LayerLevel
from gdsfactory.types import Layer
from ubc.config import PATH


@pydantic.dataclasses.dataclass(frozen=True)
class LayerMap:
    WG: Layer = (1, 0)
    DEVREC = (68, 0)
    LABEL = (10, 0)
    PORT = (1, 10)
    FLOORPLAN = (99, 0)


LAYER = LayerMap()


def get_layer_stack_ubc(thickness_nm: float = 220.0) -> LayerStack:
    """Returns generic LayerStack"""
    return LayerStack(
        layers=[
            LayerLevel(
                name="core",
                gds_layer=1,
                thickness_nm=thickness_nm,
                zmin_nm=0.0,
                material="si",
            ),
            LayerLevel(
                name="core2",
                gds_layer=31,
                thickness_nm=thickness_nm,
                zmin_nm=0.0,
                material="si",
            ),
        ]
    )


LAYER_STACK = get_layer_stack_ubc()


strip = gf.partial(
    gf.cross_section.strip, layer_cladding=LAYER.DEVREC, layers_cladding=(LAYER.DEVREC,)
)


@pydantic.dataclasses.dataclass
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


SIMULATION_SETTINGS = SimulationSettings()


@pydantic.dataclasses.dataclass
class Tech:
    name: str = "ubc"
    layer: LayerMap = LAYER

    layer_stack: LayerStack = LAYER_STACK
    sparameters_path: str = str(PATH.sparameters)


TECH = Tech()
