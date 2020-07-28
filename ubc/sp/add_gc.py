import pp
from gdslib import plot_circuit
from simphony.library import siepic
from simphony.netlist import Subcircuit


def add_gc_te(circuit, gc=siepic.ebeam_gc_te1550):
    """ add input and output gratings

    Args:
        circuit: needs to have `input` and `output` pins
        gc: grating coupler
    """
    c = Subcircuit(f"{circuit}_gc")
    gc = pp.call_if_func(gc)
    c.add([(gc, "gci"), (gc, "gco"), (circuit, "circuit")])
    c.connect_many(
        [("gci", "n1", "circuit", "input"), ("gco", "n1", "circuit", "output")]
    )

    # c.elements["circuit"].pins["input"] = "input_circuit"
    # c.elements["circuit"].pins["output"] = "output_circuit"
    c.elements["gci"].pins["n2"] = "input"
    c.elements["gco"].pins["n2"] = "output"

    return c


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ubc.cm.mzi import mzi

    c1 = mzi()
    c2 = add_gc_te(c1)
    plot_circuit(c2)
    plt.show()
