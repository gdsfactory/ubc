import pp
from ubc.bend90 import bend90
from ubc.waveguide import waveguide


@pp.cell
def ring(
    wg_width=0.5, gap=0.2, bend_radius=5, length_x=4, length_y=2,
):
    """Single bus ring made of a ring coupler (cb: bottom)
    connected with two vertical waveguides (wl: left, wr: right)
    two bends (bl, br) and horizontal waveguide (wg: top)

    Args:
        wg_width: waveguide width
        gap: gap between for coupler
        bend_radius: for the bend and coupler
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
    coupler = pp.c.coupler_ring(
        length_x=length_x, wg_width=wg_width, bend_radius=bend_radius
    )

    return pp.c.ring_single(
        wg_width=0.5,
        gap=gap,
        length_x=length_x,
        bend_radius=bend_radius,
        length_y=length_y,
        coupler=coupler,
        waveguide=waveguide,
        bend=bend90,
    )


if __name__ == "__main__":
    import ubc

    c = ring()
    cc = ubc.add_gc(c, optical_routing_type=1)
    print(c.ports)
    pp.show(cc)
