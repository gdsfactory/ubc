"""Wavelength dependent effective index model.

based on https://github.com/SiEPIC-Kits/SiEPIC_Photonics_Package
"""

import numpy as np

wavelength_start = 1500e-9
wavelength_stop = 1600e-9
resolution = 0.001
wavelength_um = (
    np.linspace(
        wavelength_start,
        wavelength_stop,
        round((wavelength_stop - wavelength_start) * 1e9 / resolution),
    )
    * 1e6
)


def neff(wavelength_um=wavelength_um, n1=2.4, n2=-1.0, n3=0.0, wavelength0_um=1.55):
    """Waveguide model neff."""
    w = wavelength_um
    w0 = wavelength0_um
    return n1 + n2 * (w - w0) + n3 * (w - w0) ** 2


def beta(wavelength_um=wavelength_um, alpha=1e-3, neff=neff, n1=2.4, n2=-1, n3=0):
    """Propagation constant.

    Args:
        wavelength_um: in um.
        alpha: propagation loss [micron^-1] constant.
    """
    if callable(neff):
        neff = neff(wavelength_um, n1=n1, n2=n2, n3=n3)

    return 2 * np.pi * neff / wavelength_um - 1j * alpha / 2 * np.ones(
        np.size(wavelength_um)
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.plot(wavelength_um, neff(wavelength_um))
    plt.show()
