import klayout.db as kl
import pp
import ubc
from pp.pack import pack

size = (605, 410)


def add_floorplan(c, size=(605, 410), layer=ubc.LAYER.FLOORPLAN):
    c << pp.c.rectangle(size=size, layer=layer)


def test_mask2():
    """ spirals for extracting waveguide loss """
    N = 15
    bend_radius = 15

    e = []
    e.append(
        ubc.spiral_te(
            N=N,
            bend_radius=bend_radius,
            y_straight_inner_top=0,
            x_inner_length_cutback=0,
        )
    )
    e.append(
        ubc.spiral_te(
            N=N,
            bend_radius=bend_radius,
            y_straight_inner_top=30,
            x_inner_length_cutback=85,
        )
    )
    c = pack(e)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_2"
    add_floorplan(m)
    pp.write_gds(m, precision=1e-9)
    m.show()


def test_mask1():
    e = [ubc.add_gc(ubc.waveguide())]
    e += [ubc.mzi_te(delta_length=dl) for dl in [9.32, 93.19]]
    e += [
        ubc.ring_single_te(bend_radius=12, gap=gap, length_x=coupling_length)
        for gap in [0.2]
        for coupling_length in [2.5, 4.5, 6.5]
    ]

    e += [
        ubc.dbr_te(w0=w0, dw=dw)
        for w0 in [0.5]
        for dw in [50e-3, 100e-3, 150e-3, 200e-3]
    ]
    e += [ubc.add_gc(ubc.crossing_te_ring())]
    e += [ubc.add_gc(ubc.crossing_te_ring(with_dut=False))]

    c = pack(e, max_size=size)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_1"
    add_floorplan(m)
    pp.write_gds(m, precision=1e-9)
    m.show()


def test_mask3():
    """contains mirror cavities and structures inside a resonator"""
    e = [ubc.add_gc(ubc.crossing_te())]
    e += [ubc.add_gc(ubc.dcate(), optical_routing_type=1)]
    e += [ubc.add_gc(ubc.dcbte())]
    e += [ubc.add_gc(ubc.y_splitter(), optical_routing_type=1)]
    e += [ubc.add_gc(ubc.y_adiabatic(), optical_routing_type=1)]
    c = pack(e)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_3"
    add_floorplan(m)
    pp.write_gds(m, precision=1e-9)
    m.show()


if __name__ == "__main__":
    # test_mask1()
    # test_mask2()
    test_mask3()
