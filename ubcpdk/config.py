"""Loads a default_config from this file.

Can overwrite config with an optional `config.yml` file in the current working directory.
"""

__all__ = ["PATH", "CONFIG"]

import pathlib

from gdsfactory.config import CONF

cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
module = pathlib.Path(__file__).parent.absolute()
repo = module.parent

CONFIG = CONF

if not hasattr(CONFIG, "username"):
    CONFIG.username = "JoaquinMatres"


class Path:
    module = module
    repo = repo
    samples = module / "samples"
    data = samples / "data"
    gds = module / "gds"

    gds_ant = gds / "ANT"
    gds_ebeam = gds / "EBeam"
    gds_beta = gds / "EBeam_Beta"
    gds_dream = gds / "EBeam_Dream"
    gds_single = gds / "EBeam_SiN"

    sparameters = repo / "sparameters"
    interconnect_cml_path = module / "simulation" / "lumerical" / "EBeam.cml"
    modes = repo / "modes"
    mask = module / "samples" / "build"
    lyp_yaml = module / "layers.yaml"

    mzi = data / "mzi"
    mzi1 = mzi / "ZiheGao_MZI1_272_Scan1.mat"
    mzi3 = mzi / "ZiheGao_MZI2_271_Scan1.mat"
    mzi1 = mzi / "ZiheGao_MZI3_270_Scan1.mat"
    mzi4 = mzi / "ZiheGao_MZI4_269_Scan1.mat"
    mzi5 = mzi / "ZiheGao_MZI5_268_Scan1.mat"
    mzi6 = mzi / "ZiheGao_MZI6_267_Scan1.mat"
    mzi8 = mzi / "ZiheGao_MZI8_266_Scan1.mat"
    mzi11 = mzi / "ZiheGao_MZI1_273_Scan1.mat"
    mzi17 = mzi / "ZiheGao_MZI17_265_Scan1.mat"
    ring = data / "ring"
    ring_te_r3_g100 = ring / "LukasC_RingDoubleTER3g100_1498.mat"
    ring_te_r3_g150 = ring / "LukasC_RingDoubleTER3g150_1497.mat"
    ring_te_r10_g50 = ring / "LukasC_RingDoubleTER10g50_1496.mat"
    ring_te_r10_g100 = ring / "LukasC_RingDoubleTER10g100_1495.mat"
    ring_te_r10_g150 = ring / "LukasC_RingDoubleTER10g150_1494.mat"
    ring_te_r10_g200 = ring / "LukasC_RingDoubleTER10g200_1493.mat"
    ring_tm_r30_g150 = ring / "LukasC_RingDoubleTMR30g150_1492.mat"
    ring_tm_r30_g200 = ring / "LukasC_RingDoubleTMR30g200_1491.mat"
    ring_tm_r30_g250 = ring / "LukasC_RingDoubleTMR30g250_1490.mat"
    dbr = data / "bragg"
    lyp = module / "klayout" / "tech" / "layers.lyp"
    lyt = module / "klayout" / "tech" / "tech.lyt"
    layers_yaml = module / "layers.yaml"
    tech = module / "klayout" / "tech"


PATH = Path()


if __name__ == "__main__":
    print(PATH.sparameters)
