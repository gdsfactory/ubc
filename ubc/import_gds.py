import pathlib

import pp
from ubc.layers import LAYER

cwd = pathlib.Path(__file__).parent.absolute()
gds = cwd / "gds"


layer = LAYER.WG
port_width = 0.5


def guess_port_orientaton(position, name, label, n):
    """ we assume that ports with x<0 are inputs (orientation=180deg)
    and ports with x>0 are outputs
    """
    p = position
    if "gc" in name:
        return 0
    if label == "opt1":
        return 180
    if p[1] > 0 and "crossing" in name:
        return 90
    if p[1] < 0 and "crossing" in name:
        return 270
    if n == 4 and label in ["opt1", "opt2"]:
        return 180
    if n == 4 and label in ["opt3", "opt4"]:
        return 0
    if p[0] <= 0:
        return 180
    return 0


def import_gds(gdsname):
    """ import gds from SIEPIC PDK
    """
    c = pp.import_gds(gds / f"{gdsname}.gds")

    n = 0
    for label in c.get_labels():
        if label.text.startswith("opt"):
            n += 1

    for label in c.get_labels():
        if label.text.startswith("opt"):
            port = pp.Port(
                name=label.text,
                midpoint=label.position,
                width=port_width,
                orientation=guess_port_orientaton(
                    position=label.position, name=gdsname, label=label.text, n=n,
                ),
                layer=layer,
            )
            c.add_port(port)

    return c


if __name__ == "__main__":
    gdsname = "ebeam_y_1550"
    c = import_gds(gdsname)
    print(c.ports)
    pp.show(c)
