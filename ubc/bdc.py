import pp
from ubc.import_gds import import_gds


@pp.cell
def bdc():
    """Broadband directional coupler."""
    c = import_gds("ebeam_bdc_te1550")
    return c


if __name__ == "__main__":
    c = bdc()
    c.show()
