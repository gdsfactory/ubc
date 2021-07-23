import pandas as pd
import pydantic

import pp
import pp.sp as sp

from ubc.tech import LAYER_STACK, LayerStack
from ubc.config import PATH


@pydantic.validate_arguments
def write_sparameters(
    component: pp.Component,
    layer_stack: LayerStack = LAYER_STACK,
    session=None,
    run: bool = True,
    overwrite: bool = False,
    dirpath=PATH.sparameters,
    **settings,
) -> pd.DataFrame:
    """Writes Sparameters in Lumerical FDTD."""
    return sp.write(
        component=component,
        layer_stack=layer_stack,
        session=session,
        run=run,
        overwrite=overwrite,
        dirpath=dirpath,
        **settings,
    )


if __name__ == "__main__":
    import ubc.components as pdk

    c = pdk.straight()
    df = write_sparameters(component=c)
