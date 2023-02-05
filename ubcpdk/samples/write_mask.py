"""Sample mask for the course."""

from typing import Tuple
from pathlib import Path
from omegaconf import OmegaConf
import gdsfactory as gf

from ubcpdk.tech import LAYER
from ubcpdk.config import PATH


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


if __name__ == "__main__":
    pass
