import pp
from ubc.import_gds import import_gds


def y_adiabatic():
    """Y junction adiabatic."""
    c = import_gds("ebeam_y_adiabatic")
    return c


if __name__ == "__main__":
    import ubc

    c = y_adiabatic()
    cc = ubc.add_gc(c, optical_routing_type=1)
    print(c.ports)
    pp.show(cc)
