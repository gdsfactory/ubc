# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Component FDTD simulations
#
# Thanks to the GDSFactory plugin you can directly run simulations in different FDTD solvers.
#
# See [tutorial](https://gdsfactory.github.io/gdsfactory/plugins_fdtd.html)

# ## Tidy3d
#
# You can read about the [tidy3d gdsfactory plugin](https://gdsfactory.github.io/gdsfactory/notebooks/plugins/tidy3d/00_tidy3d.html)

# +
import gdsfactory.simulation as sim
import gdsfactory.simulation.gtidy3d as gt

import ubcpdk
import ubcpdk.components as pdk

# -

c = pdk.ebeam_y_1550()
c

sp = gt.write_sparameters(c)

sp.keys()

sim.plot.plot_sparameters(sp)

sim.plot.plot_loss1x2(sp)

sim.plot.plot_imbalance1x2(sp)

# ## Lumerical FDTD
#
# You can write the [Sparameters](https://en.wikipedia.org/wiki/Scattering_parameters) for all components in the UBC `ubcpdk.components` PDK using lumerical FDTD plugin in gdsfactory

# To run simulations uncomment the following lines

import gdsfactory.simulation as sim
import ubcpdk.components as pdk

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


# +
# sp = ls.read.read_sparameters_lumerical(component=ubcpdk.components.straight())

# +
# sim.plot_sparameters(sp)
# -

# ## MEEP FDTD
#
# Meep in an open source FDTD library developed at MIT.
# See [docs](https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/) and [code](https://github.com/NanoComp/meep).
#
# You can use the gdsfactory meep plugin to run simulation using meep. You can run examples with `resolution=20` so they run fast.
#
# The resolution is in pixels/um so you need to run with at least `resolution=100` for 1/100 um/pixel (10 nm/ pixel).

import gdsfactory as gf
import gdsfactory.simulation.gmeep as gm

c = ubcpdk.components.straight(length=3)
c

df = gm.write_sparameters_meep_1x1(component=c, run=False)

df = gm.write_sparameters_meep_1x1(component=c, run=True)

gm.plot.plot_sparameters(df)

gm.plot.plot_sparameters(df, logscale=False)

c = ubcpdk.components.ebeam_y_1550()
c

df = gm.write_sparameters_meep(component=c, run=False)  # lr stands for left-right ports

df = gm.write_sparameters_meep(
    gf.components.coupler_ring(), xmargin=3, ymargin_bot=3, run=False
)  # lr stands for left-right ports
