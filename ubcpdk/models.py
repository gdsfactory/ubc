from __future__ import annotations

from functools import partial

from gplugins.sax.models import (
    attenuator,
    bend,
    coupler,
    grating_coupler,
    mmi1x2,
    mmi2x2,
    phase_shifter,
)
from gplugins.sax.models import straight as _straight

nm = 1e-3


straight = partial(_straight, wl0=1.55, neff=2.4, ng=4.2)
bend_euler = partial(bend, loss=0.03)

gc_te1550 = partial(grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.55)
gc_te1550_broadband = partial(grating_coupler, loss=6, bandwidth=50 * nm, wl0=1.55)
gc_tm1550 = partial(grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.55)

gc_te1310_broadband = partial(grating_coupler, loss=6, bandwidth=50 * nm, wl0=1.31)
gc_te1310 = partial(grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.31)


models = dict(
    attenuator=attenuator,
    bend_euler=bend,
    coupler=coupler,
    mmi1x2=mmi1x2,
    mmi2x2=mmi2x2,
    phase_shifter=phase_shifter,
    straight=straight,
    taper=straight,
    gc_te1550=gc_te1550,
    gc_te1550_broadband=gc_te1550_broadband,
    gc_tm1550=gc_tm1550,
    gc_te1310_broadband=gc_te1310_broadband,
    gc_te1310=gc_te1310,
)


if __name__ == "__main__":
    import gplugins.sax as gs

    gs.plot_model(grating_coupler)
    # gs.plot_model(coupler)
