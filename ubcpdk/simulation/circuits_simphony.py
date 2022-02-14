from functools import partial

import gdsfactory.simulation.simphony as gs
from simphony.library import siepic


def ebeam_y_1550():
    c = siepic.ebeam_y_1550()
    c.pins = ("o1", "o2", "o3")
    return c


def ebeam_bdc_te1550():
    c = siepic.ebeam_bdc_te1550()
    c.pins = ("o1", "o2", "o4", "o3")
    return c


def ebeam_dc_halfring_straight():
    c = siepic.ebeam_dc_halfring_straight()
    c.pins = ("o1", "o2", "o4", "o3")
    return c


def ebeam_dc_te1550():
    c = siepic.ebeam_dc_te1550()
    c.pins = ("o1", "o2", "o4", "o3")
    return c


def ebeam_gc_te1550():
    c = siepic.ebeam_gc_te1550()
    c.pins = ("o1", "o2")
    return c


def ebeam_terminator_te1550():
    c = siepic.ebeam_terminator_te1550()
    c.pins = ("o1",)
    return c


mzi = partial(
    gs.components.mzi,
    splitter=ebeam_y_1550,
)


model_factory = dict(
    ebeam_y_1550=ebeam_y_1550,
    ebeam_bdc_te1550=ebeam_bdc_te1550,
    ebeam_dc_halfring_straight=ebeam_dc_halfring_straight,
    ebeam_dc_te1550=ebeam_dc_te1550,
    ebeam_gc_te1550=ebeam_gc_te1550,
    ebeam_terminator_te1550=ebeam_terminator_te1550,
)

circuit_factory = dict(mzi=mzi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gdsfactory.simulation.simphony.plot_circuit import plot_circuit

    c = mzi()
    plot_circuit(c)
    plt.show()
