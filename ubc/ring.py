import pp
from ubc.bend90 import bend90
from ubc.waveguide import waveguide


@pp.cell
def ring(
    width=0.5, gap=0.2, radius=5, length_x=4, length_y=2,
):
    """Single bus ring made of a ring coupler (cb: bottom)
    connected with two vertical waveguides (wl: left, wr: right)
    two bends (bl, br) and horizontal waveguide (wg: top)

    Args:
        width: waveguide width
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical waveguide length

    .. code::

          bl-wt-br
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x
    """
    coupler = pp.c.coupler_ring(length_x=length_x, width=width, radius=radius)

    return pp.c.ring_single(
        width=0.5,
        gap=gap,
        length_x=length_x,
        radius=radius,
        length_y=length_y,
        coupler=coupler,
        waveguide=waveguide,
        bend=bend90,
    )


if __name__ == "__main__":
    from ubc.add_gc import add_gc

    c = ring()
    cc = add_gc(c, optical_routing_type=1)
    print(c.ports)
    cc.show()
