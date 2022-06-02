"""Technology definitions

- LayerStack
- cross_sections (xs_)
- constants (WIDTH, CLADDING_OFFSET ...)

TODO: make sure routes use cross_section
"""
import sys

from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_section_factories
from gdsfactory.tech import LayerStack, LayerLevel
from gdsfactory.types import Layer
import gdsfactory.simulation as sim
import gdsfactory.simulation.lumerical as lumerical
from gdsfactory.add_pins import add_pins_bbox_siepic as add_pins_bbox_siepic_10nm
from gdsfactory.add_pins import add_pins_siepic as add_pins_siepic_10nm

from ubcpdk.config import PATH

nm = 1e-3

add_pins_siepic = gf.partial(add_pins_siepic_10nm, pin_length=100 * nm)
add_pins_bbox_siepic = gf.partial(add_pins_bbox_siepic_10nm, pin_length=100 * nm)


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

## TODO: update_source will generate a layers.py, so use that instead
# if os.path.exists(PATH.tech / "layers.py"):
#     from .klayout.tech.layers import LayerMap as LayerMapUbc
#
# else:


class LayerMapUbc(BaseModel):
    WG: Layer = (1, 0)
    WG2: Layer = (31, 0)
    DEVREC: Layer = (68, 0)
    LABEL: Layer = (10, 0)
    PORT: Layer = (1, 10)  # PinRec
    PORTE: Layer = (1, 11)  # PinRecM
    FLOORPLAN: Layer = (99, 0)

    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (66, 0)
    LABEL_INSTANCE: Layer = (66, 0)

    class Config:
        frozen = True
        extra = "forbid"


LAYER = LayerMapUbc()


def get_layer_stack_ubc(thickness: float = 220 * nm) -> LayerStack:
    """Returns generic LayerStack"""
    ## TODO: Translate xml in lumerical process file
    return LayerStack(
        layers=dict(
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
    )


class Tech(BaseModel):
    name: str = "ubc"
    layer: LayerMapUbc = LAYER

    fiber_array_spacing: float = 250.0
    WG = {"width": 0.5}
    DEVREC = {"width": 0.5}


TECH = Tech()


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


strip_wg_simulation_info = dict(
    model="ebeam_wg_integral_1550",
    library="Design kits/ebeam",
    layout_model_property_pairs=(
        # (layout_property_name, interconnect_property_name)
        ("length", "wg_length", 1e-6),
        ("width", "wg_width", 1e-6),
    ),
    layout_model_port_pairs=(("o1", "port 1"), ("o2", "port 2")),
    spice_params=["wg_length", "wg_width"],
    component_type=["optical"],
    properties=(("annotate", False),),
)

get_sparameters_data_lumerical = gf.partial(
    sim.get_sparameters_data_lumerical,
    layer_stack=LAYER_STACK,
    dirpath=PATH.sparameters,
)


strip_pins = gf.partial(
    gf.cross_section.strip,
    layer=LAYER.WG,
    width=TECH.WG["width"],
    info=strip_wg_simulation_info,
    add_pins=add_pins_siepic,
)
strip = gf.partial(
    gf.cross_section.strip,
    layer=LAYER.WG,
    width=TECH.WG["width"],
    info=strip_wg_simulation_info,
    add_pins=add_pins_siepic,
)
strip_no_pins = gf.partial(
    gf.cross_section.cross_section,
    layer=LAYER.WG,
    width=TECH.WG["width"],
)

cross_sections = get_cross_section_factories(sys.modules[__name__])


__all__ = ("add_pins_siepic", "add_pins_bbox_siepic")


if __name__ == "__main__":
    c = gf.c.straight(cross_section=strip)
    c.show(show_ports=False)
