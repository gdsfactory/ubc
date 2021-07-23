import pp
from pp.component import Component
from pp.tech import Library
from ubc.import_gds import import_gds

from ubc.components.add_fiber_array import add_fiber_array
from ubc.components.grating_couplers import (
    gc_te1550,
    gc_te1550_broadband,
    gc_te1310,
    gc_tm1550,
)
from ubc.components.straight import straight
from ubc.components.dbr import dbr_cavity, dbr
from ubc.components.crossing import crossing, ring_with_crossing


def dc_broadband_te() -> Component:
    """Broadband directional coupler TE1550 50/50 power."""
    return import_gds("ebeam_bdc_te1550")


def dc_broadband_tm() -> Component:
    """Broadband directional coupler TM1550 50/50 power."""
    return import_gds("ebeam_bdc_tm1550")


def dc_adiabatic() -> Component:
    """Adiabatic directional coupler TE1550 50/50 power."""
    return import_gds("ebeam_adiabatic_te1550")


def y_adiabatic() -> Component:
    """Adiabatic Y junction TE1550 50/50 power."""
    return import_gds("ebeam_y_adiabatic")


def y_splitter() -> Component:
    """Y junction TE1550 50/50 power."""
    return import_gds("ebeam_y_1550")


def spiral(**kwargs):
    return pp.c.spiral_external_io(**kwargs)


LIBRARY = Library(name="ubc")
LIBRARY.register(
    [
        add_fiber_array,
        crossing,
        dbr,
        dbr_cavity,
        dc_adiabatic,
        dc_broadband_te,
        dc_broadband_tm,
        gc_te1310,
        gc_te1550,
        gc_te1550_broadband,
        gc_tm1550,
        ring_with_crossing,
        spiral,
        straight,
        y_adiabatic,
        y_splitter,
    ]
)


__all__ = list(LIBRARY.factory.keys())


if __name__ == "__main__":
    # c = straight_no_pins()
    # c = add_fiber_array(component=c)
    # c = gc_tm1550()
    # print(c.get_ports_array())
    # print(c.ports.keys())
    c = straight()
    c = add_fiber_array(component=c)
    c.pprint()
    c.show()
