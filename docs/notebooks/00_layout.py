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

# # Layout
#
# All UBC ubcpdk.components cells are conveniently combined into the ubcpdk.components module.

# +
import gdsfactory as gf

import ubcpdk

# -

# ## Fixed Component cells
#
# Most `ubcpdk` components are imported from GDS files as fixed cells.

ubcpdk.components.ebeam_crossing4()

ubcpdk.components.ebeam_swg_edgecoupler()

ubcpdk.components.ebeam_bdc_te1550()

ubcpdk.components.ebeam_adiabatic_te1550()

ubcpdk.components.ebeam_y_adiabatic()

ubcpdk.components.ebeam_y_1550()

# ## Parametric Component PCells
#
# You can also define cells adapted from gdsfactory generic pdk.

ubcpdk.components.straight(length=2)

ubcpdk.components.bend_euler(radius=5)

ubcpdk.components.ring_with_crossing()

ubcpdk.components.dbr()

ubcpdk.components.spiral()

ubcpdk.components.mzi_heater()

ubcpdk.components.ring_single_heater()

# ## Components with grating couplers
#
# To test your devices you can add grating couplers. Both for single fibers and for fiber arrays.

splitter = ubcpdk.components.ebeam_y_1550(decorator=gf.port.auto_rename_ports)
mzi = gf.components.mzi(splitter=splitter)
mzi

component_fiber_array = ubcpdk.components.add_fiber_array(component=mzi)
component_fiber_array

c = ubcpdk.components.ring_single_heater()
c = ubcpdk.components.add_fiber_array_pads_rf(c)
c

c = ubcpdk.components.mzi_heater()
c = ubcpdk.components.add_fiber_array_pads_rf(c, optical_routing_type=2)
c

# ## 3D rendering

scene = c.to_3d()
scene.show()
