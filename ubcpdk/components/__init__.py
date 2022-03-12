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
    mzi,
    ring_single,
    spiral,
    y_adiabatic,
    y_splitter,
    bend_euler,
    crossing,
    ring_with_crossing,
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
from ubcpdk.components.straight import (
    straight,
)

factory = dict(
    add_fiber_array=add_fiber_array,
    bend_euler=bend_euler,
    crossing=crossing,
    dbr=dbr,
    dbr_cavity=dbr_cavity,
    dc_adiabatic=dc_adiabatic,
    dc_broadband_te=dc_broadband_te,
    dc_broadband_tm=dc_broadband_tm,
    gc_te1310=gc_te1310,
    gc_te1550=gc_te1550,
    gc_te1550_broadband=gc_te1550_broadband,
    gc_tm1550=gc_tm1550,
    ring_with_crossing=ring_with_crossing,
    spiral=spiral,
    straight=straight,
    y_adiabatic=y_adiabatic,
    y_splitter=y_splitter,
)


__all__ = [
    "add_fiber_array",
    "bend_euler",
    "cells",
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
    "straight",
    "y_adiabatic",
    "y_splitter",
]
