import gdslib as gl
from gdslib import autoname
from gdslib.types import ModelFactory
import gdsfactory as gf
from simphony.library import siepic
from simphony.netlist import Subcircuit


@autoname
def mzi(
    length_y: float = 1.0,
    delta_length: float = 100.0,
    length_x: float = 10.0,
    y_model_factory: ModelFactory = siepic.ebeam_y_1550,
    waveguide: ModelFactory = siepic.ebeam_wg_integral_1550,
) -> Subcircuit:
    """Mzi circuit model

    Args:
        length_y: vertical length for both and top arms (um)
        delta_length: bottom arm extra length
        length_x: horizontal length for both and top arms (um)
        waveguide: waveguide_model

    .. code::

                   __Lx__
                  |      |
                  Ly     Lyr (not a parameter)
                  |      |
        splitter==|      |==combiner
                  |      |
                  Ly     Lyr (not a parameter)
                  |      |
                  | delta_length/2
                  |      |
                  |__Lx__|



    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.c.mzi(length_y=0.1, delta_length=0, length_x=10)
      gf.plotgds(c)


    .. plot::
        :include-source:

        import ubcpdk
        import gdslib as gl

        c = ubcpdk.circuits.mzi()
        gl.plot_circuit(c)

    """
    y = gf.call_if_func(y_model_factory)
    wg_long = waveguide(length=(2 * length_y + delta_length + length_x) * 1e-6)
    wg_short = waveguide(length=(2 * length_y + length_x) * 1e-6)

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
    gl.simphony.plot_circuit(m)
    plt.show()
