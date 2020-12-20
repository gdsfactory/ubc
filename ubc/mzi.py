import pp
from ubc.bend90 import bend90
from ubc.waveguide import waveguide
from ubc.y_splitter import y_splitter


@pp.cell
def mzi(
    delta_length=100,
    length_y: float = 4.0,
    length_x: float = 0.1,
    waveguide=waveguide,
    bend90=bend90,
    splitter=y_splitter,
    **kwargs
):
    """Mzi interferometer.

    Args:
        delta_length: bottom arm vertical extra length
        length_y: vertical length for both and top arms
        length_x: horizontal length
        bend_radius: 10.0
        bend90: bend_circular
        waveguide: waveguide function
        waveguide_vertical: waveguide
        splitter: splitter function
        combiner: combiner function
        with_splitter: if False removes splitter
        pins: add pins cell and child cells
        combiner_settings: settings dict for combiner function
        splitter_settings: settings dict for splitter function

    .. code::

                   __Lx__
                  |      |
                  Ly     Lyr
                  |      |
        splitter==|      |==combiner
                  |      |
                  Ly     Lyr
                  |      |
                 DL/2   DL/2
                  |      |
                  |__Lx__|



    """
    c = pp.c.mzi(
        delta_length=delta_length,
        waveguide=waveguide,
        bend90=bend90,
        splitter=splitter,
        length_x=length_x,
        length_y=length_y,
        **kwargs
    )
    return c


if __name__ == "__main__":
    c = mzi(delta_length=100)
    pp.show(c)
