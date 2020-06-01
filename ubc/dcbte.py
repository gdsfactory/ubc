import pp
from ubc.import_gds import import_gds


def dcbte():
    """ directional coupler broadband 1550 te """
    c = import_gds("ebeam_bdc_te1550")
    return c


if __name__ == "__main__":
    c = dcbte()
    pp.show(c)
