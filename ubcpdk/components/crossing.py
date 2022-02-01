from gdsfactory.component import Component
from gdsfactory.components.ring_single_dut import ring_single_dut

from ubcpdk.import_gds import import_gds


def crossing() -> Component:
    """TE waveguide crossing."""
    return import_gds("ebeam_crossing4.gds")


def ring_with_crossing(**kwargs) -> Component:
    return ring_single_dut(component=crossing(), **kwargs)


if __name__ == "__main__":
    c = ring_with_crossing()
    c.show()
