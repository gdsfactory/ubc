import pp
import ubc
from ubc.pdk import PDK

L = 1.55 / 4 / 2 / 2.44


@pp.cell
def dbr_te(w0=0.5, dw=0.1, n=600, l1=L, l2=L, **kwargs):
    """Cavity with a DBR
    """
    mirror = pp.c.dbr(
        w1=w0 - dw / 2,
        w2=w0 + dw / 2,
        n=n,
        l1=l1,
        l2=l2,
        waveguide_function=PDK.waveguide,
        **kwargs,
    )
    cavity = pp.c.cavity(component=mirror)
    return ubc.add_gc(
        component=cavity,
        component_name=f"dbr-{int(dw*1e3)}-{int(w0*1e3)}",
        optical_routing_type=0,
        fanout_length=0,
    )


if __name__ == "__main__":
    c = dbr_te()
    c.show()
