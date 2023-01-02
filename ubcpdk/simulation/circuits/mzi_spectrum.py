"""MZI spectrum model.

based on https://github.com/SiEPIC-Kits/SiEPIC_Photonics_Package
"""

import numpy as np
from ubcpdk.simulation.circuits.waveguide import beta, neff, wavelength_um


def mzi_spectrum(
    L1_um,
    L2_um,
    wavelength_um=wavelength_um,
    beta=beta,
    alpha=1e-3,
    neff=neff,
    n1=2.4,
    n2=-1,
    n3=0,
):
    """Returns MZI spectrum.

    Args:
        L1_um.
        L2_um.
        wavelength_um.
        beta: propagation constant.
    """
    if callable(beta):
        beta = beta(wavelength_um, neff=neff, alpha=alpha, n1=n1, n2=n2, n3=n3)

    return 0.25 * np.abs(np.exp(-1j * beta * L1_um) + np.exp(-1j * beta * L2_um)) ** 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # plt.plot(wavelength_um, mzi_spectrum(100, 110))
    plt.plot(wavelength_um, 10 * np.log10(mzi_spectrum(L1_um=40, L2_um=255)))
    plt.show()
