import pp
from pp.pack import pack
from ubc import PDK
from ubc.tech import LAYER

size = (605, 410)


def add_floorplan(c, size=(605, 410), layer=LAYER.FLOORPLAN):
    c << pp.c.rectangle(size=size, layer=layer)


def add_gc(component, **kwargs):
    c = PDK.add_fiber_array(component, **kwargs)
    c.name = f"{component.name}_te"
    return c


def test_mask2():
    """ spirals for extracting waveguide loss """
    N = 15
    bend_radius = 15

    e = []
    e.append(
        PDK.add_fiber_array(
            PDK.spiral(
                N=N,
                bend_radius=bend_radius,
                y_straight_inner_top=0,
                x_inner_length_cutback=0,
            )
        )
    )
    e.append(
        PDK.add_fiber_array(
            PDK.spiral(
                N=N,
                bend_radius=bend_radius,
                y_straight_inner_top=30,
                x_inner_length_cutback=85,
            )
        )
    )

    c = pack(e)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_2"
    add_floorplan(m)
    pp.write_gds(m, precision=1e-9)
    m.show()
    return m


def test_mask1():
    e = [add_gc(PDK.waveguide())]
    e += [add_gc(PDK.mzi(delta_length=dl)) for dl in [9.32, 93.19]]
    e += [
        add_gc(PDK.ring_single(radius=12, gap=gap, length_x=coupling_length))
        for gap in [0.2]
        for coupling_length in [2.5, 4.5, 6.5]
    ]

    e += [
        add_gc(PDK.dbr_cavity(w0=w0, dw=dw))
        for w0 in [0.5]
        for dw in [50e-3, 100e-3, 150e-3, 200e-3]
    ]
    e += [add_gc(PDK.ring_with_crossing())]
    e += [add_gc(PDK.ring_with_crossing(with_dut=False))]

    c = pack(e, max_size=size)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_1"
    add_floorplan(m)
    pp.write_gds(m, precision=1e-9)
    m.show()
    return m


def test_mask3():
    """contains mirror cavities and structures inside a resonator"""
    e = [add_gc(PDK.crossing())]
    e += [add_gc(PDK.dc_adiabatic(), optical_routing_type=1)]
    e += [add_gc(PDK.dc_broadband())]
    e += [add_gc(PDK.y_splitter(), optical_routing_type=1)]
    e += [add_gc(PDK.y_adiabatic(), optical_routing_type=1)]
    c = pack(e)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_3"
    add_floorplan(m)
    pp.write_gds(m, precision=1e-9)
    m.show()
    return m


if __name__ == "__main__":
    # m1 = test_mask1()
    # m2 = test_mask2()
    m3 = test_mask3()
