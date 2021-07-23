import pp
from pp import Component


L = 1.55 / 4 / 2 / 2.44


def straight(waveguide: pp.types.StrOrDict = "strip", **kwargs):
    return pp.c.straight(waveguide=waveguide, **kwargs)


@pp.cell_with_validator
def dbr(
    w0: float = 0.5, dw: float = 0.1, n: int = 600, l1: float = L, l2: float = L
) -> Component:
    return pp.components.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        straight=straight,
    )


def dbr_cavity(**kwargs) -> Component:
    return pp.c.cavity(component=dbr(**kwargs))


if __name__ == "__main__":
    c = dbr_cavity()
    c.show()
