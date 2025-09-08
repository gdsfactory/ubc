"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

sax.set_port_naming_strategy("optical")

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


def gc_te1550(
    wl: float | NDArray = 1.55,
    loss: float = 6,
    bandwidth: float = 35 * nm,
    wl0: float = 1.55,
) -> sax.SDict:
    """Grating coupler model for TE polarization at 1550 nm with custom parameters.
    Args:
        wl: Wavelength in micrometers. Can be a scalar or array for multi-wavelength
            simulations. Defaults to 1.55 μm.
        loss: Coupling loss in dB. Represents the insertion loss of the grating
            coupler. Typical values are around 6 dB.
        bandwidth: 3 dB bandwidth in micrometers. Defines the wavelength range
            over which the coupler operates effectively. Typical values are
            around 35 nm (0.035 μm).
        wl0: Center wavelength in micrometers. The wavelength at which the
            coupler is optimized. Defaults to 1.55 μm.
    """
    return sm.grating_coupler(wl=wl, loss=loss, bandwidth=bandwidth, wl0=wl0)


def gc_te1550_broadband(
    wl: float | NDArray = 1.55,
    loss: float = 6,
    bandwidth: float = 50 * nm,
    wl0: float = 1.55,
) -> sax.SDict:
    """Grating coupler model for TE polarization at 1550 nm with custom parameters.
    Args:
        wl: Wavelength in micrometers. Can be a scalar or array for multi-wavelength
            simulations. Defaults to 1.55 μm.
        loss: Coupling loss in dB. Represents the insertion loss of the grating
            coupler. Typical values are around 6 dB.
        bandwidth: 3 dB bandwidth in micrometers. Defines the wavelength range
            over which the coupler operates effectively. Typical values are
            around 50 nm (0.05 μm).
        wl0: Center wavelength in micrometers. The wavelength at which the
            coupler is optimized. Defaults to 1.55 μm.
    """
    return sm.grating_coupler(wl=wl, loss=loss, bandwidth=bandwidth, wl0=wl0)


def gc_tm1550(
    wl: float | NDArray = 1.55,
    loss: float = 6,
    bandwidth: float = 35 * nm,
    wl0: float = 1.55,
) -> sax.SDict:
    """Grating coupler model for TM polarization at 1550 nm with custom parameters.
    Args:
        wl: Wavelength in micrometers. Can be a scalar or array for multi-wavelength
            simulations. Defaults to 1.55 μm.
        loss: Coupling loss in dB. Represents the insertion loss of the grating
            coupler. Typical values are around 6 dB.
        bandwidth: 3 dB bandwidth in micrometers. Defines the wavelength range
            over which the coupler operates effectively. Typical values are
            around 35 nm (0.035 μm).
        wl0: Center wavelength in micrometers. The wavelength at which the
            coupler is optimized. Defaults to 1.55 μm.
    """
    return sm.grating_coupler(wl=wl, loss=loss, bandwidth=bandwidth, wl0=wl0)


def gc_te1310_broadband(
    wl: float | NDArray = 1.31,
    loss: float = 6,
    bandwidth: float = 50 * nm,
    wl0: float = 1.31,
) -> sax.SDict:
    """Grating coupler model for TE polarization at 1310 nm with custom parameters.
    Args:
        wl: Wavelength in micrometers. Can be a scalar or array for multi-wavelength
            simulations. Defaults to 1.31 μm.
        loss: Coupling loss in dB. Represents the insertion loss of the grating
            coupler. Typical values are around 6 dB.
        bandwidth: 3 dB bandwidth in micrometers. Defines the wavelength range
            over which the coupler operates effectively. Typical values are
            around 50 nm (0.05 μm).
        wl0: Center wavelength in micrometers. The wavelength at which the
            coupler is optimized. Defaults to 1.31 μm.
    """
    return sm.grating_coupler(wl=wl, loss=loss, bandwidth=bandwidth, wl0=wl0)


def gc_te1310(
    wl: float | NDArray = 1.31,
    loss: float = 6,
    bandwidth: float = 35 * nm,
    wl0: float = 1.31,
) -> sax.SDict:
    """Grating coupler model for TE polarization at 1310 nm with custom parameters.
    Args:
        wl: Wavelength in micrometers. Can be a scalar or array for multi-wavelength
            simulations. Defaults to 1.31 μm.
        loss: Coupling loss in dB. Represents the insertion loss of the grating
            coupler. Typical values are around 6 dB.
        bandwidth: 3 dB bandwidth in micrometers. Defines the wavelength range
            over which the coupler operates effectively. Typical values are
            around 35 nm (0.035 μm).
        wl0: Center wavelength in micrometers. The wavelength at which the
            coupler is optimized. Defaults to 1.31 μm.
    """
    return sm.grating_coupler(wl=wl, loss=loss, bandwidth=bandwidth, wl0=wl0)


################
# MMIs
################
def mmi1x2(
    wl: float | NDArray = 1.55,
    fwhm: float = 0.2,
    loss_dB: float = 0.3,
) -> sax.SDict:
    """1x2 MMI model for 1550 nm with custom parameters.

    Args:
        wl: Wavelength in micrometers. Can be a scalar or array for multi-wavelength
            simulations. Defaults to 1.55 μm.
        fwhm: Full-width half-maximum bandwidth in micrometers. Defines the
            wavelength range over which the MMI operates effectively. Typical
            values are around 0.2 μm (200 nm).
        loss_dB: Excess loss in dB. Represents additional insertion loss of the
            MMI beyond ideal splitting. Typical values are around 0.3 dB.
    """
    return sm.mmi1x2(wl=wl, fwhm=fwhm, loss_dB=loss_dB)


def mmi2x2(
    wl: float | NDArray = 1.55,
    fwhm: float = 0.2,
    loss_dB: float = 0.3,
) -> sax.SDict:
    """2x2 MMI model for 1550 nm with custom parameters.
    Args:
        wl: Wavelength in micrometers. Can be a scalar or array for multi-wavelength
            simulations. Defaults to 1.55 μm.
        fwhm: Full-width half-maximum bandwidth in micrometers. Defines the
            wavelength range over which the MMI operates effectively. Typical
            values are around 0.2 μm (200 nm).
        loss_dB: Excess loss in dB. Represents additional insertion loss of the
            MMI beyond ideal splitting. Typical values are around 0.3 dB.
    """
    return sm.mmi2x2(wl=wl, fwhm=fwhm, loss_dB=loss_dB)


def ebeam_y_1550(
    wl: float | NDArray = 1.55,
    fwhm: float = 0.2,
    loss_dB: float = 0.3,
) -> sax.SDict:
    """Y-branch model for 1550 nm with custom parameters.
    Args:
        wl: Wavelength in micrometers. Can be a scalar or array for multi-wavelength
            simulations. Defaults to 1.55 μm.
        fwhm: Full-width half-maximum bandwidth in micrometers. Defines the
            wavelength range over which the Y-branch operates effectively. Typical
            values are around 0.2 μm (200 nm).
        loss_dB: Excess loss in dB. Represents additional insertion loss of the
            Y-branch beyond ideal splitting. Typical values are around 0.3 dB.
    """
    return sm.mmi1x2(wl=wl, fwhm=fwhm, loss_dB=loss_dB)


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
        # Skip get_models itself and private functions
        if name == "get_models" or name.startswith("_"):
            continue
        if not callable(func):
            continue
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            continue
        # Check for sax.SDict return type (case-insensitive)
        return_anno = str(sig.return_annotation)
        if "sdict" in return_anno.lower():
            models[name] = func
    return models


_models = get_models()

if __name__ == "__main__":
    print(sorted(_models.keys()))
