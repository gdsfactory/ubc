import pp
from ubc.import_gds import import_gds


def y_adiabatic():
    """Y junction adiabatic."""
    c = import_gds("ebeam_y_adiabatic")
    return c


if __name__ == "__main__":
    from ubc.add_gc import add_gc
    from ubc.pdk import PDK

    c = y_adiabatic()
    # cc = add_gc(c, optical_routing_type=1)
    cc = PDK.add_fiber_array(c)
    print(c.ports)
    cc.show()
