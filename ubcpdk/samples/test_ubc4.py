from functools import partial

import gdsfactory as gf

from ubcpdk import PDK, cells
from ubcpdk.tech import LAYER

size = (440, 470)
add_gc = cells.add_fiber_array


@gf.cell
def EBeam_JoaquinMatres_4() -> gf.Component:
    """MZI interferometers."""
    mzi = partial(gf.components.mzi, splitter=cells.ebeam_y_1550)
    mzis = [mzi(delta_length=delta_length) for delta_length in [10, 40, 100]]
    mzis_gc = [cells.add_fiber_array(mzi) for mzi in mzis]

    mzis = [cells.mzi_heater(delta_length=delta_length) for delta_length in [40]]
    mzis_heater_gc = [
        cells.add_fiber_array_pads_rf(mzi, orientation=90) for mzi in mzis
    ]

    e = mzis_gc + mzis_heater_gc
    c = gf.Component()
    _ = c << gf.pack(e, max_size=size, spacing=2)[0]
    _ = c << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return c


def test_ubc4() -> None:
    c = EBeam_JoaquinMatres_4()
    assert c


if __name__ == "__main__":
    PDK.activate()
    c = EBeam_JoaquinMatres_4()
    c.show()  # show in klayout
