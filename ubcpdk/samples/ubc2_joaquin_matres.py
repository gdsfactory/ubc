"""Sample mask for the course."""

from typing import Tuple
from pathlib import Path
from omegaconf import OmegaConf
import gdsfactory as gf

import ubcpdk
from ubcpdk.tech import LAYER
from ubcpdk.config import PATH

size = (605, 410)
add_gc = ubcpdk.components.add_fiber_array


def write_mask_gds_with_metadata(m) -> Tuple[Path, Path]:
    """Returns"""
    gdspath = PATH.mask / f"{m.name}.gds"
    m.write_gds_with_metadata(gdspath=gdspath)

    labels_path = gf.labels.write_labels.write_labels_gdstk(
        gdspath=gdspath, layer_label=LAYER.LABEL
    )
    metadata_path = gdspath.with_suffix(".yml")
    test_metadata_path = gdspath.with_suffix(".tp.yml")
    mask_metadata = OmegaConf.load(metadata_path)

    tm = gf.labels.merge_test_metadata(
        labels_path=labels_path, mask_metadata=mask_metadata
    )
    test_metadata_path.write_text(OmegaConf.to_yaml(tm))
    return m, tm


def test_mask1():
    """spirals for extracting straight waveguide loss"""
    mzi = gf.partial(gf.components.mzi, splitter=ubcpdk.components.ebeam_y_1550)
    mzis = [mzi(delta_length=delta_length) for delta_length in [10, 100]]
    mzis_with_gc = [add_gc(mzi) for mzi in mzis]

    c = gf.pack(mzis_with_gc)
    m = c[0]
    m.name = "EBeam_JoaquinMatres_1"

    fp = m << gf.components.compass(size=size, layer=LAYER.FLOORPLAN)
    fp.move((size[0] / 2, size[1] / 2))
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    m1, tm1 = test_mask1()
    m1.show()
