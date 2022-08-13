"""MPB mode simulations."""

import gdsfactory as gf
import gdsfactory.simulation.modes as gm
from ubcpdk.config import PATH


nm = 1e-3

find_modes_waveguide = gf.partial(
    gm.find_modes_waveguide,
    wg_width=500 * nm,
    wg_thickness=220 * nm,
    slab_thickness=0 * nm,
    resolution=20,
    nmodes=4,
    cache=PATH.modes,
)


find_neff_vs_width = gf.partial(
    gm.find_neff_vs_width, cache=PATH.modes, filepath="find_neff_vs_width.csv"
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = find_neff_vs_width()
    gm.plot_neff_vs_width(df)
    plt.show()
