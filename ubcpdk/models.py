"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

nm = 1e-3


def straight(
    wl: float | NDArray = 1.55,
    length: float = 10.0,
    neff: float = 2.4,
    ng: float = 4.2,
    wl0: float = 1.55,
    loss_dB_cm: float = 3.0,
) -> sax.SDict:
    """Dispersive straight waveguide model.

    ```
    in0             out0
     o1 =========== o2
    ```

    Args:
        wl: The wavelength in micrometers.
        wl0: The center wavelength used for dispersion calculation.
        neff: The Effective refractive index at the center wavelength.
        ng: The Group refractive index at the center wavelength.
        length: The length of the waveguide in micrometers.
        loss_dB_cm: The Propagation loss in dB/cm.
    """
    return sm.straight(
        wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm
    )


def bend_euler(
    wl: float | NDArray = 1.55,
    wl0: float = 1.55,
    neff: float = 2.4,
    ng: float = 4.2,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
) -> sax.SDict:
    """Simple waveguide bend model.

    ```
              out0
              o2

             /
         __.'
    o1
    in0
    ```

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            multi-wavelength simulations. Defaults to 1.55 μm.
        wl0: Reference wavelength in micrometers used for dispersion calculation.
            This is typically the design wavelength where neff is specified.
            Defaults to 1.55 μm.
        neff: Effective refractive index at the reference wavelength. This value
            represents the fundamental mode effective index and determines the
            phase velocity. Defaults to 2.34 (typical for silicon).
        ng: Group refractive index at the reference wavelength. Used to calculate
            chromatic dispersion: ng = neff - lambda * d(neff)/d(lambda).
            Typically ng > neff for normal dispersion. Defaults to 3.4.
        length: Physical length of the waveguide in micrometers. Determines both
            the total phase accumulation and loss. Defaults to 10.0 μm.
        loss_dB_cm: Propagation loss in dB/cm. Includes material absorption,
            scattering losses, and other loss mechanisms. Set to 0.0 for
            lossless modeling. Defaults to 0.0 dB/cm.
    """
    return sm.bend(
        wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm
    )


###################
# grating couplers
###################
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


def straight_heater_metal(
    wl: float = 1.55,
    neff: float = 2.34,
    voltage: float = 0,
    vpi: float = 1.0,  # Voltage required for π-phase shift
    length: float = 10,
    loss: float = 0.0,
) -> sax.SDict:
    """Returns simple phase shifter model.

    Args:
        wl: wavelength.
        neff: effective index.
        voltage: applied voltage.
        vpi: voltage required for a π-phase shift.
        length: length.
        loss: loss.
    """
    # Calculate additional phase shift due to applied voltage.
    deltaphi = (voltage / vpi) * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
            ("l_e1", "r_e1"): 0.0,
            ("l_e2", "r_e2"): 0.0,
            ("l_e3", "r_e3"): 0.0,
            ("l_e4", "r_e4"): 0.0,
        }
    )


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return a dictionary of all models in this module."""
    models = {}
    for name, func in list(globals().items()):
        if not callable(func):
            continue
        _func = func
        while isinstance(_func, partial):
            _func = _func.func
        try:
            sig = inspect.signature(_func)
        except ValueError:
            print(f"Could not get signature for {name}")
            continue
        if str(sig.return_annotation).lower().split(".")[-1] == "sdict":
            models[name] = func
    return models


_models = get_models()

if __name__ == "__main__":
    print(sorted(_models.keys()))
