"""Sample mask for the course."""
from functools import partial
from pathlib import Path

import gdsfactory as gf
from omegaconf import OmegaConf

import ubcpdk
from ubcpdk.config import PATH
from ubcpdk.tech import LAYER

size = (440, 470)
add_gc = ubcpdk.components.add_fiber_array
pack = partial(
    gf.pack, max_size=size, add_ports_prefix=False, add_ports_suffix=False, spacing=2
)


def write_mask_gds_with_metadata(m) -> Path:
    """Returns"""
    gdspath = PATH.mask / f"{m.name}.gds"
    m.write_gds_with_metadata(gdspath=gdspath)
    metadata_path = gdspath.with_suffix(".yml")
    OmegaConf.load(metadata_path)
    gf.labels.write_labels.write_labels_gdstk(
        gdspath=gdspath, layer_label=LAYER.LABEL, debug=True
    )
    # test_metadata_path = gdspath.with_suffix(".tp.yml")

    # tm = gf.labels.merge_test_metadata(
    #     labels_path=labels_path, mask_metadata=mask_metadata
    # )
    # test_metadata_path.write_text(OmegaConf.to_yaml(tm))
    return gdspath
