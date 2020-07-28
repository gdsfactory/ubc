import gdslib as gl
import pp
from simphony.library import siepic
from simphony.netlist import Subcircuit


@pp.autoname
def mzi(
    L0=1,
    L1=100,
    L2=10,
    y_model_factory=siepic.ebeam_y_1550,
    wg=siepic.ebeam_wg_integral_1550,
):
    """ Mzi circuit model

    Args:
        L0 (um): vertical length for both and top arms
        L1 (um): bottom arm extra length, delta_length = 2*L1
        L2 (um): L_top horizontal length

    .. code::

               __L2__
               |      |
               L0     L0r
               |      |
     splitter==|      |==recombiner
               |      |
               L0     L0r
               |      |
               L1     L1
               |      |
               |__L2__|


    .. plot::
      :include-source:

      import pp

      c = pp.c.mzi(L0=0.1, L1=0, L2=10)
      pp.plotgds(c)


    .. plot::
        :include-source:

        import ubc
        import gdslib as gl

        c = ubc.sp.mzi()
        gl.plot_circuit(c)

    """
    y = pp.call_if_func(y_model_factory)
    wg_long = wg(length=(2 * L0 + 2 * L1 + L2) * 1e-6)
    wg_short = wg(length=(2 * L0 + L2) * 1e-6)

    # Create the circuit, add all individual instances
    circuit = Subcircuit("mzi")
    circuit.add(
        [
            (y, "splitter"),
            (y, "recombiner"),
            (wg_long, "wg_long"),
            (wg_short, "wg_short"),
        ]
    )
    circuit.elements["splitter"].pins = ("in1", "out1", "out2")
    circuit.elements["recombiner"].pins = ("out1", "in2", "in1")

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("splitter", "out1", "wg_long", "n1"),
            ("splitter", "out2", "wg_short", "n1"),
            ("recombiner", "in1", "wg_long", "n2"),
            ("recombiner", "in2", "wg_short", "n2"),
        ]
    )
    circuit.elements["splitter"].pins["in1"] = "input"
    circuit.elements["recombiner"].pins["out1"] = "output"
    return circuit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = mzi()
    gl.plot_circuit(m)
    plt.show()
