"""
based on https://github.com/SiEPIC-Kits/SiEPIC_Photonics_Package
"""

import numpy as np
from ubc.cm.waveguide import beta, wavelength_um


def mzi(L1, L2, wavelength_um=wavelength_um):
    return (
        0.25
        * np.abs(
            np.exp(-1j * beta(wavelength_um) * L1)
            + np.exp(-1j * beta(wavelength_um) * L2)
        )
        ** 2
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # plt.plot(wavelength_um, mzi(100, 110))
    plt.plot(wavelength_um, 10 * np.log10(mzi(100, 110)))
    plt.show()
