"""Sample mask for the edx course Q1 2023."""

import gdsfactory as gf

import ubcpdk
import ubcpdk.components as pdk

# from ubcpdk.tech import add_labels_to_ports_optical
from ubcpdk.tech import LAYER
from ubcpdk.samples.write_mask import write_mask_gds_with_metadata, add_gc, pack, size


add_labels_to_ports_optical = gf.add_labels.add_labels_to_ports_optical


def test_mask1():
    """Add DBR cavities."""
    e = [add_gc(ubcpdk.components.straight())]
    e += [add_gc(gf.components.mzi(delta_length=dl)) for dl in [9.32, 93.19]]
    e += [
        add_gc(gf.components.ring_single(radius=12, gap=gap, length_x=coupling_length))
        for gap in [0.2]
        for coupling_length in [2.5, 4.5, 6.5]
    ]

    e += [
        ubcpdk.components.dbr_cavity_te(w0=w0, dw=dw)
        for w0 in [0.5]
        for dw in [50e-3, 100e-3, 150e-3, 200e-3]
    ]
    e += [add_gc(ubcpdk.components.ring_with_crossing())]
    e += [
        add_gc(
            ubcpdk.components.ring_with_crossing(port_name="o2", with_component=False)
        )
    ]

    c = pack(e)
    m = c[0]
    add_labels_to_ports_optical(m)
    m.name = "EBeam_JoaquinMatres_11"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask2():
    """spirals for extracting straight waveguide loss"""
    N = 15
    radius = 15

    e = [
        ubcpdk.components.add_fiber_array(
            component=ubcpdk.components.spiral(
                N=N,
                radius=radius,
                y_straight_inner_top=0,
                x_inner_length_cutback=0,
                info=dict(does=["spiral", "te1550"]),
            )
        )
    ]

    e.append(
        ubcpdk.components.add_fiber_array(
            component=ubcpdk.components.spiral(
                N=N,
                radius=radius,
                y_straight_inner_top=30,
                x_inner_length_cutback=85,
            )
        )
    )

    c = pack(e)

    m = c[0]
    m.name = "EBeam_JoaquinMatres_12"
    add_labels_to_ports_optical(m)
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask3():
    """contains mirror cavities and structures inside a resonator"""
    e = []
    e += [add_gc(ubcpdk.components.ebeam_crossing4())]
    e += [add_gc(ubcpdk.components.ebeam_adiabatic_te1550(), optical_routing_type=1)]
    e += [add_gc(ubcpdk.components.ebeam_bdc_te1550())]
    e += [add_gc(ubcpdk.components.ebeam_y_1550(), optical_routing_type=1)]
    e += [add_gc(ubcpdk.components.ebeam_y_adiabatic_tapers(), optical_routing_type=1)]
    e += [
        add_gc(ubcpdk.components.straight(), component_name=f"straight_{i}")
        for i in range(2)
    ]
    c = pack(e)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_13"
    # add_labels_to_ports_optical(m)
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask4():
    """MZI interferometers."""
    mzi = gf.partial(gf.components.mzi, splitter=ubcpdk.components.ebeam_y_1550)
    mzis = [mzi(delta_length=delta_length) for delta_length in [10, 40, 100]]
    mzis_gc = [pdk.add_fiber_array(mzi) for mzi in mzis]

    mzis = [pdk.mzi_heater(delta_length=delta_length) for delta_length in [40]]
    mzis_heater_gc = [
        pdk.add_fiber_array_pads_rf(mzi, optical_routing_type=2) for mzi in mzis
    ]

    c = pack(mzis_gc + mzis_heater_gc)
    m = c[0]
    add_labels_to_ports_optical(m)
    m.name = "EBeam_JoaquinMatres_14"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask5():
    """Ring resonators."""

    rings = [pdk.ring_single_heater(length_x=length_x) for length_x in [4, 6]]
    rings_gc = [pdk.add_fiber_array_pads_rf(ring) for ring in rings]

    c = pack(rings_gc)
    m = c[0]
    add_labels_to_ports_optical(m)
    m.name = "EBeam_JoaquinMatres_15"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    gf.clear_cache()

    # m, tm = test_mask1()  # dbr and mzi
    # m, tm = test_mask2() # spirals
    m, tm = test_mask3()  # coupler and crossing
    # m, tm = test_mask4()  # heated mzis
    # m, tm = test_mask5()  # heated rings
    m.show()
