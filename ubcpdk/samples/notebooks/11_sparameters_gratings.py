# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Grating coupler FDTD simulations
#
# You can also expand the planar component simulations to simulate an out-of-plane grating coupler.
#
# You can run grating coupler simulations in 2D to save time, and for accuracy you can also run them in 3D
#
# ## tidy3d

# %%
import gplugins.gtidy3d as gt
import matplotlib.pyplot as plt
import numpy as np

import ubcpdk.components as pdk
from ubcpdk.config import PATH

c = pdk.gc_te1550()
c.plot()

# %%
fiber_angle_deg = -31
s = gt.get_simulation_grating_coupler(
    c, is_3d=False, fiber_angle_deg=fiber_angle_deg, fiber_xoffset=0
)
f = gt.plot_simulation(s)

# %%
offsets = np.arange(-15, 6, 5)
offsets

# %%
jobs = [
    dict(
        component=c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xoffset=fiber_xoffset,
        dirpath=PATH.sparameters,
    )
    for fiber_xoffset in offsets
]
sps = gt.write_sparameters_grating_coupler_batch(jobs)


# %%
def log(x):
    return 20 * np.log10(x)


# %%
for offset in offsets:
    sp = gt.write_sparameters_grating_coupler(
        c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xoffset=offset,
        dirpath=PATH.sparameters,
    )
    plt.plot(
        sp["wavelengths"],
        20 * np.log10(np.abs(sp["o2@0,o1@0"])),
        label=str(offset),
    )

plt.xlabel("wavelength (um")
plt.ylabel("Transmission (dB)")
plt.title("transmission vs fiber xoffset (um)")
plt.legend()

# %%
help(gt.write_sparameters_grating_coupler)
