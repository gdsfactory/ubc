from typing import Tuple
from omegaconf import OmegaConf
from pathlib import Path
import gdsfactory as gf

import ubc
from ubc.tech import LAYER
from ubc.config import PATH

floorplan_size = (605, 410)
add_gc = ubc.components.add_fiber_array


def write_mask_gds_with_metadata(m) -> Tuple[Path, Path]:
    """Returns"""
    gdspath = PATH.mask / f"{m.name}.gds"
    m.write_gds_with_metadata(gdspath=gdspath)

    labels_path = gf.mask.write_labels(gdspath=gdspath, layer_label=LAYER.LABEL)
    metadata_path = gdspath.with_suffix(".yml")
    test_metadata_path = gdspath.with_suffix(".tp.yml")
    mask_metadata = OmegaConf.load(metadata_path)

    tm = gf.mask.merge_test_metadata(
        labels_path=labels_path, mask_metadata=mask_metadata
    )
    test_metadata_path.write_text(OmegaConf.to_yaml(tm))
    return m, tm


def test_mask2():
    """spirals for extracting straight waveguide loss"""
    N = 15
    radius = 15

    e = []
    e.append(
        ubc.components.add_fiber_array(
            component=ubc.components.spiral(
                N=N,
                radius=radius,
                y_straight_inner_top=0,
                x_inner_length_cutback=0,
                info=dict(does=["spiral", "te1550"]),
            )
        )
    )
    e.append(
        ubc.components.add_fiber_array(
            component=ubc.components.spiral(
                N=N,
                radius=radius,
                y_straight_inner_top=30,
                x_inner_length_cutback=85,
            )
        )
    )

    c = gf.pack(e)

    m = c[0]
    m.name = "EBeam_JoaquinMatres_2"
    m << gf.c.rectangle(size=floorplan_size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask1():
    e = [add_gc(ubc.components.straight())]
    e += [add_gc(gf.components.mzi(delta_length=dl)) for dl in [9.32, 93.19]]
    e += [
        add_gc(gf.components.ring_single(radius=12, gap=gap, length_x=coupling_length))
        for gap in [0.2]
        for coupling_length in [2.5, 4.5, 6.5]
    ]

    e += [
        add_gc(ubc.components.dbr_cavity(w0=w0, dw=dw))
        for w0 in [0.5]
        for dw in [50e-3, 100e-3, 150e-3, 200e-3]
    ]
    e += [add_gc(ubc.components.ring_with_crossing())]
    e += [add_gc(ubc.components.ring_with_crossing(with_component=False))]

    c = gf.pack(e, max_size=floorplan_size)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_1"
    m << gf.c.rectangle(size=floorplan_size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


def test_mask3():
    """contains mirror cavities and structures inside a resonator"""
    e = [add_gc(ubc.components.crossing())]
    e += [add_gc(ubc.components.dc_adiabatic(), optical_routing_type=1)]
    e += [add_gc(ubc.components.dc_broadband_te())]
    e += [add_gc(ubc.components.y_splitter(), optical_routing_type=1)]
    e += [add_gc(ubc.components.y_adiabatic(), optical_routing_type=1)]
    c = gf.pack(e)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_3"
    m << gf.c.rectangle(size=floorplan_size, layer=LAYER.FLOORPLAN)
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    m, tm = test_mask1()
    m, tm = test_mask2()
    m, tm = test_mask3()
    m.show()
