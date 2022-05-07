from pydantic import BaseModel
from gdsfactory.types import Layer


class LayerMap(BaseModel):
    Si_p6nm: Layer = (31, 0)
    Deep: Layer = (201, 0)
    DevRec: Layer = (68, 0)
    Errors: Layer = (999, 0)
    FbrTgt: Layer = (81, 0)
    FloorPlan: Layer = (99, 0)
    Lumerical: Layer = (733, 0)
    M1_heater: Layer = (11, 0)
    M2_router: Layer = (12, 0)
    M_Open: Layer = (13, 0)
    Oxide: Layer = (6, 0)
    PinRec: Layer = (1, 10)
    PinRecM: Layer = (1, 11)
    SEM: Layer = (200, 0)
    Si: Layer = (1, 0)
    Si_Litho193nm: Layer = (1, 69)
    Text: Layer = (10, 0)
    VC: Layer = (40, 0)
    Waveguide: Layer = (1, 0)

    class Config:
        frozen = True
        extra = "forbid"


LAYER = LayerMap()
