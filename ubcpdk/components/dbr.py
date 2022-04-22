import gdsfactory as gf

from ubcpdk.components.add_fiber_array import add_fiber_array
from ubcpdk.components.generic import coupler
from ubcpdk.tech import add_pins_bbox_siepic


L = 1.55 / 4 / 2 / 2.44


@gf.cell
def dbr(
    w0: float = 0.5, dw: float = 0.1, n: int = 600, l1: float = L, l2: float = L
) -> gf.Component:
    straight_no_pins = gf.partial(gf.components.straight, cross_section="strip_no_pins")

    return gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        straight=straight_no_pins,
        decorator=add_pins_bbox_siepic,
    )


def dbr_cavity(**kwargs) -> gf.Component:
    return gf.components.cavity(component=dbr(**kwargs), coupler=coupler)


def dbr_cavity_te(component="dbr_cavity", **kwargs) -> gf.Component:
    component = gf.get_component(component, **kwargs)
    return add_fiber_array(component)


if __name__ == "__main__":
    # c = dbr_cavity(n=10)
    c = dbr_cavity_te(n=10)
    c.show()
