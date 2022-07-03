from functools import partial

import gdsfactory.simulation.simphony as gs
from simphony.libraries import siepic


def ebeam_y_1550(**kwargs):
    c = siepic.YBranch(**kwargs)
    c.rename_pins("o1", "o2", "o3")
    return c


def ebeam_bdc_te1550(**kwargs):
    c = siepic.BidirectionalCoupler(**kwargs)
    c.rename_pins("o1", "o2", "o4", "o3")
    return c


def ebeam_dc_halfring_straight(**kwargs):
    c = siepic.HalfRing(**kwargs)
    c.rename_pins("o1", "o2", "o4", "o3")
    return c


def ebeam_dc_te1550(**kwargs):
    c = siepic.DirectionalCoupler(**kwargs)
    c.rename_pins("o1", "o2", "o4", "o3")
    return c


def ebeam_gc_te1550(**kwargs):
    c = siepic.GratingCoupler(**kwargs)
    c.rename_pins("o1", "o2")
    return c


def ebeam_terminator_te1550(**kwargs):
    c = siepic.Terminator(**kwargs)
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
