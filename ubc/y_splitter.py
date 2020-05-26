import pp
from pp.add_pins import add_pins
from ubc.import_gds import import_gds


@pp.autoname
def y_splitter():
    c = import_gds("ebeam_y_1550")
    add_pins(c)
    return c


if __name__ == "__main__":
    c = y_splitter()
    pp.show(c)
