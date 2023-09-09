"""Sample mask for the edx course Q1 2023."""

from functools import partial
from pathlib import Path

import gdsfactory as gf

import ubcpdk
import ubcpdk.components as pdk
from ubcpdk import tech
from ubcpdk.cutback_2x2 import cutback_2x2
from ubcpdk.samples.write_mask import add_gc, pack, size, write_mask_gds_with_metadata
from ubcpdk.tech import LAYER


def test_mask1() -> Path:
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
    m.name = "EBeam_JoaquinMatres_11"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask2() -> Path:
    """spirals for extracting straight waveguide loss"""
    N = 12
    radius = 10

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
                y_straight_inner_top=0,
                x_inner_length_cutback=185,
            )
        )
    )

    c = pack(e)

    m = c[0]
    m.name = "EBeam_JoaquinMatres_12"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask3() -> Path:
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
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask4() -> Path:
    """MZI interferometers."""
    mzi = partial(gf.components.mzi, splitter=ubcpdk.components.ebeam_y_1550)
    mzis = [mzi(delta_length=delta_length) for delta_length in [10, 40, 100]]
    mzis_gc = [pdk.add_fiber_array(mzi) for mzi in mzis]

    mzis = [pdk.mzi_heater(delta_length=delta_length) for delta_length in [40]]
    mzis_heater_gc = [
        pdk.add_fiber_array_pads_rf(mzi, orientation=90, optical_routing_type=2)
        for mzi in mzis
    ]

    c = pack(mzis_gc + mzis_heater_gc)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_14"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask5() -> Path:
    """Ring resonators."""
    rings = [pdk.ring_single_heater(length_x=length_x) for length_x in [4, 6]]
    rings = [gf.functions.rotate180(ring) for ring in rings]
    rings_gc = [pdk.add_fiber_array_pads_rf(ring) for ring in rings]

    c = pack(rings_gc)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_15"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask6() -> Path:
    """Splitters 1x2."""
    mmis = []
    mmis += [
        gf.components.cutback_splitter(
            component=pdk.ebeam_y_adiabatic_tapers, cols=1, rows=7
        )
    ]
    mmis += [gf.components.cutback_splitter(component=pdk.ebeam_y_1550, cols=6, rows=7)]
    mmis_gc = [pdk.add_fiber_array(mmi, optical_routing_type=1) for mmi in mmis]

    c = pack(mmis_gc)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_16"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask7() -> Path:
    """Splitters 2x2."""
    # mmi2x2_with_sbend = partial(
    #     gf.components.mmi2x2_with_sbend,
    #     decorator=tech.add_pins_bbox_siepic_remove_layers,
    # )

    mmi2x2_with_sbend = tech.add_pins_bbox_siepic_remove_layers(
        gf.components.mmi2x2_with_sbend().flatten()
    )
    mmi2x2_with_sbend.name = "mmi2x2_with_sbend"

    mmis = []
    mmis += [cutback_2x2(component=pdk.ebeam_bdc_te1550, cols=3)]
    mmis += [cutback_2x2(component=mmi2x2_with_sbend, cols=6)]

    mmis += [mmi2x2_with_sbend]
    mmis += [pdk.ebeam_bdc_te1550()]

    mmis_gc = [
        pdk.add_fiber_array(component=mmi, optical_routing_type=1) for mmi in mmis
    ]
    c = pack(mmis_gc)

    m = c[0]
    m.name = "EBeam_JoaquinMatres_17"
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    # m = test_mask1()  # dbr and mzi
    # m = test_mask2()  # spirals
    # m = test_mask3()  # coupler and crossing
    # m = test_mask4()  # heated mzis
    # m = test_mask5()  # heated rings
    # m = test_mask6()  # 1x2 mmis
    m = test_mask7()  # 2x2mmis
    gf.show(m)
    # c = partial(
    #     gf.components.mmi2x2_with_sbend,
    #     decorator=tech.add_pins_bbox_siepic_remove_layers,
    # )()
    # c.show()
