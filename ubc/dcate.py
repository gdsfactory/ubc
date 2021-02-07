import pp
import ubc
from ubc.import_gds import import_gds


def dcate():
    """ directional coupler adiabatic 1550 te """
    c = import_gds("ebeam_adiabatic_te1550")
    return c


if __name__ == "__main__":
    from ubc.add_gc import add_gc

    c = dcate()
    cc = add_gc(c, optical_routing_type=1)
    print(c.ports)
    cc.show()
