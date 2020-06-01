import pp
from pp.components.ring_single_dut import ring_single_dut
from pp.ports import deco_rename_ports
from ubc.import_gds import import_gds


@deco_rename_ports
@pp.autoname
def crossing_te():
    c = import_gds("ebeam_crossing4")
    return c


def crossing_te_ring(**kwargs):
    return ring_single_dut(component=crossing_te(), **kwargs)


if __name__ == "__main__":
    # c = crossing_te()
    c = crossing_te_ring(with_dut=False)
    print(c.ports)
    pp.show(c)
