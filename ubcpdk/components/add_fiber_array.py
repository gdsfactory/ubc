import gdsfactory as gf
from gdsfactory.dft.siepic import add_fiber_array_siepic

from ubcpdk.components.grating_couplers import gc_te1550


add_fiber_array = gf.partial(
    add_fiber_array_siepic,
    gc_port_name="opt1",
    grating_coupler=gc_te1550,
)


if __name__ == "__main__":
    c = add_fiber_array()
    c.show()
