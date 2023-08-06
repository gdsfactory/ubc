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
# # Component FDTD simulations
#
# Thanks to the GDSFactory plugin you can directly run simulations in different FDTD solvers.
#
# See [tutorial](https://gdsfactory.github.io/gdsfactory/plugins_fdtd.html)

# %% [markdown]
# ## Tidy3d
#
# You can read about the [tidy3d gdsfactory plugin](https://gdsfactory.github.io/gdsfactory/notebooks/plugins/tidy3d/00_tidy3d.html)

# %%
import gplugins as sim
import gplugins.gtidy3d as gt

import ubcpdk
import ubcpdk.components as pdk
from ubcpdk.config import PATH


# %%
c = pdk.ebeam_y_1550()
c.plot()

# %%
sp = gt.write_sparameters(c)

# %%
sp.keys()

# %%
sim.plot.plot_sparameters(sp)

# %%
sim.plot.plot_loss1x2(sp)

# %%
sim.plot.plot_imbalance1x2(sp)

# %% [markdown]
# ## Lumerical FDTD
#
# You can write the [Sparameters](https://en.wikipedia.org/wiki/Scattering_parameters) for all components in the UBC `ubcpdk.components` PDK using lumerical FDTD plugin in gdsfactory

# %% [markdown]
# To run simulations uncomment the following lines

# %%
import gplugins as sim
import ubcpdk.components as pdk

# %%
for f in [
    pdk.bend_euler,
    pdk.coupler,
    pdk.coupler_ring,
    pdk.ebeam_y_1550,
    pdk.ebeam_crossing4,
]:
    component = f()
    component.plot()
    # ls.write_sparameters_lumerical(component=component)


# %%
# sp = ls.read.read_sparameters_lumerical(component=ubcpdk.components.straight())

# %%
# sim.plot_sparameters(sp)

# %% [markdown]
# ## MEEP FDTD
#
# Meep in an open source FDTD library developed at MIT.
# See [docs](https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/) and [code](https://github.com/NanoComp/meep).
#
# You can use the gdsfactory meep plugin to run simulation using meep. You can run examples with `resolution=20` so they run fast.
#
# The resolution is in pixels/um so you need to run with at least `resolution=100` for 1/100 um/pixel (10 nm/ pixel).

# %%
import gdsfactory as gf
import gplugins.gmeep as gm

# %%
c = ubcpdk.components.straight(length=3)
c.plot()

# %%
df = gm.write_sparameters_meep_1x1(component=c, run=False)

# %%
df = gm.write_sparameters_meep_1x1(component=c, run=True, dirpath=PATH.sparameters)

# %%
gm.plot.plot_sparameters(df)

# %%
gm.plot.plot_sparameters(df, logscale=False)

# %%
c = ubcpdk.components.ebeam_y_1550()
c

# %%
df = gm.write_sparameters_meep(component=c, run=False)  # lr stands for left-right ports
