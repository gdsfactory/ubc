from functools import partial

import gdsfactory.simulation.simphony as gs
from simphony.libraries import siepic


def ebeam_y_1550():
    c = siepic.YBranch()
    c.rename_pins("o1", "o2", "o3")
    return c


def ebeam_bdc_te1550():
    c = siepic.BidirectionalCoupler()
    c.rename_pins("o1", "o2", "o4", "o3")
    return c


def ebeam_dc_halfring_straight():
    c = siepic.HalfRing()
    c.rename_pins("o1", "o2", "o4", "o3")
    return c


def ebeam_dc_te1550():
    c = siepic.DirectionalCoupler()
    c.rename_pins("o1", "o2", "o4", "o3")
    return c


def ebeam_gc_te1550():
    c = siepic.GratingCoupler()
    c.rename_pins("o1", "o2")
    return c


def ebeam_terminator_te1550():
    c = siepic.Terminator()
    c.rename_pins(
        "o1",
    )
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
    # from gdsfactory.simulation.simphony.plot_circuit import plot_circuit
    # c = mzi()
    # plot_circuit(c)

    gs.plot_model(ebeam_gc_te1550)
