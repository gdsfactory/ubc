import gdsfactory as gf
import gdsfactory.simulation as sim

from ubc.tech import LAYER_STACK
from ubc.config import PATH


write_sparameters = gf.partial(
    sim.write_sparameters_lumerical, dirpath=PATH.sparameters, layer_stack=LAYER_STACK
)


if __name__ == "__main__":
    import ubc.components as pdk

    c = pdk.straight()
    df = write_sparameters(component=c)
