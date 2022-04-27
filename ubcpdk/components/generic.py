import gdsfactory as gf

from ubcpdk.tech import add_pins_bbox_siepic


bend_with_pins = gf.partial(
    gf.components.bend_euler,
    cross_section="strip",
)
coupler = gf.partial(
    gf.components.coupler, decorator=add_pins_bbox_siepic, cross_section="strip"
)
coupler_ring = gf.partial(gf.components.coupler_ring, cross_section="strip")

ring_single = gf.partial(
    gf.components.ring_single,
    bend=bend_with_pins,
    coupler_ring=coupler_ring,
    cross_section="strip",
)

spiral = gf.partial(gf.components.spiral_external_io)


if __name__ == "__main__":
    # c = coupler_ring()
    # c = ring_single()
    # c = bend_with_pins()
    c = coupler()
    c.show(show_ports=False)
