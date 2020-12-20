import pp
from ubc.import_gds import import_gds


@pp.cell
def y_splitter():
    """Ysplitter."""
    c = import_gds("ebeam_y_1550")
    c.name = "Ebeam.ebeam_y_1550"
    return c


if __name__ == "__main__":
    c = y_splitter()
    print(c.ports)
    pp.show(c)
