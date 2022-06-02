import gdsfactory as gf
from ubcpdk.components import cells
from ubcpdk.components import grating_couplers

from ubcpdk.components.add_fiber_array import (
    add_fiber_array,
)
from ubcpdk.components.cells import (
    dc_adiabatic,
    dc_broadband_te,
    dc_broadband_tm,
    ebeam_dc_halfring_straight,
    ebeam_dc_te1550,
    y_adiabatic,
    y_splitter,
    bend_euler,
    straight,
    crossing,
    ring_with_crossing,
    mzi,
)
from ubcpdk.components.dbr import (
    dbr,
    dbr_cavity,
)
from ubcpdk.components.grating_couplers import (
    gc_te1310,
    gc_te1550,
    gc_te1550_broadband,
    gc_tm1550,
)

from ubcpdk.components.generic import coupler, ring_single, spiral

pad = gf.partial(gf.components.pad, layer="WG")

__all__ = [
    "add_fiber_array",
    "bend_euler",
    "straight",
    "cells",
    "coupler",
    "crossing",
    "dbr",
    "dbr_cavity",
    "dc_adiabatic",
    "dc_broadband_te",
    "dc_broadband_tm",
    "ebeam_dc_halfring_straight",
    "ebeam_dc_te1550",
    "gc_te1310",
    "gc_te1550",
    "gc_te1550_broadband",
    "gc_tm1550",
    "grating_couplers",
    "mzi",
    "ring_single",
    "ring_with_crossing",
    "spiral",
    "y_adiabatic",
    "y_splitter",
    "pad",
]
