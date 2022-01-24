import gdsfactory as gf
from ubcpdk.tech import strip


L = 1.55 / 4 / 2 / 2.44


def straight(cross_section: gf.types.CrossSectionFactory = strip, **kwargs):
    return gf.c.straight(cross_section=cross_section, **kwargs)


@gf.cell
def dbr(
    w0: float = 0.5, dw: float = 0.1, n: int = 600, l1: float = L, l2: float = L
) -> gf.Component:
    return gf.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        straight=straight,
    )


def dbr_cavity(**kwargs) -> gf.Component:
    return gf.c.cavity(component=dbr(**kwargs))


if __name__ == "__main__":
    c = dbr_cavity()
    c.show()
