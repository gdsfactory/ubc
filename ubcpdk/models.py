from __future__ import annotations

from functools import partial

import gplugins.sax.models as sm

nm = 1e-3


straight = partial(sm.straight, wl0=1.55, neff=2.4, ng=4.2)
bend_euler_sc = bend_euler = partial(sm.bend, loss=0.03)

################
# grating couplers
################
gc_te1550 = partial(sm.grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.55)
gc_te1550_broadband = partial(sm.grating_coupler, loss=6, bandwidth=50 * nm, wl0=1.55)
gc_tm1550 = partial(sm.grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.55)
gc_te1310_broadband = partial(sm.grating_coupler, loss=6, bandwidth=50 * nm, wl0=1.31)
gc_te1310 = partial(sm.grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.31)

################
# MMIs
################
mmi1x2 = partial(sm.mmi1x2, wl0=1.55, fwhm=0.2, loss_dB=0.3)
mmi2x2 = partial(sm.mmi2x2, wl0=1.55, fwhm=0.2, loss_dB=0.3)
ebeam_y_1550 = mmi1x2
coupler = sm.coupler

if __name__ == "__main__":
    import gplugins.sax as gs

    gs.plot_model(gc_te1550)
    gs.plot_model(coupler)
