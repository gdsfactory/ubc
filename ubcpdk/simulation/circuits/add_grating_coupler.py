import gdsfactory as gf
from gdslib.simphony import plot_circuit
from simphony.library import siepic
from simphony.netlist import Subcircuit


def add_grating_coupler(circuit, grating_coupler=siepic.ebeam_gc_te1550):
    """Add input and output gratings couplers.

    Args:
        circuit: needs to have `input` and `output` pins
        gc: grating coupler model function
    """
    c = Subcircuit(f"{circuit}_gc")
    gc = gf.call_if_func(grating_coupler)
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
    from mzi import mzi

    c1 = mzi()
    c2 = add_grating_coupler(c1)
    plot_circuit(c2)
    plt.show()
