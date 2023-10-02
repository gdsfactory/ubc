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
# # Layout
#
# All UBC ubcpdk.components cells are conveniently combined into the ubcpdk.components module.

# %%
import gdsfactory as gf

import ubcpdk

gf.CONF.display_type = "klayout"

# %% [markdown]
# ## Fixed Component cells
#
# Most `ubcpdk` components are imported from GDS files as fixed cells.

# %%
c = ubcpdk.components.ebeam_crossing4()
c.plot()

# %%
c = ubcpdk.components.ebeam_swg_edgecoupler()
c.plot()

# %%
c = ubcpdk.components.ebeam_bdc_te1550()
c.plot()

# %%
c = ubcpdk.components.ebeam_adiabatic_te1550()
c.plot()

# %%
c = ubcpdk.components.ebeam_y_adiabatic()
c.plot()

# %%
c = ubcpdk.components.ebeam_y_1550()
c.plot()

# %% [markdown]
# ## Parametric Component PCells
#
# You can also define cells adapted from gdsfactory generic pdk.

# %%
c = ubcpdk.components.straight(length=2)
c.plot()

# %%
c = ubcpdk.components.bend_euler(radius=5)
c.plot()

# %%
c = ubcpdk.components.ring_with_crossing()
c.plot()

# %%
c = ubcpdk.components.dbr()
c.plot()

# %%
c = ubcpdk.components.spiral()
c.plot()

# %%
c = ubcpdk.components.mzi_heater()
c.plot()

# %%
c = ubcpdk.components.ring_single_heater()
c.plot()

# %% [markdown]
# ## Components with grating couplers
#
# To test your devices you can add grating couplers. Both for single fibers and for fiber arrays.

# %%
splitter = ubcpdk.components.ebeam_y_1550(decorator=gf.port.auto_rename_ports)
mzi = gf.components.mzi(splitter=splitter)
mzi.plot()

# %%
component_fiber_array = ubcpdk.components.add_fiber_array(component=mzi)
component_fiber_array.plot()

# %%
c = ubcpdk.components.ring_single_heater()
c = ubcpdk.components.add_fiber_array_pads_rf(c)
c.plot()

# %%
c = ubcpdk.components.mzi_heater()
c = ubcpdk.components.add_fiber_array_pads_rf(c, optical_routing_type=2)
c.plot()

# %% [markdown]
# ## 3D rendering

# %%
scene = c.to_3d()
scene.show()
