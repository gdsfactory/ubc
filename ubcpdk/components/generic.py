import gdsfactory as gf
from ubcpdk.import_gds import (
    add_ports_siepic_gratings,
)


coupler = gf.partial(gf.c.coupler, decorator=add_ports_siepic_gratings)


if __name__ == "__main__":
    c = coupler()
    c.show()
