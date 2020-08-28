"""
loads a configuration from `default_config` in this file (low priority)
which can be overwritten with config.yml found in the current working directory (high priority)
"""

__all__ = ["path", "conf"]

import io
import pathlib

from omegaconf import OmegaConf

default_config = io.StringIO(
    """
username: JoaquinMatres
"""
)

cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
config_base = OmegaConf.load(default_config)
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent

try:
    config_cwd = OmegaConf.load(cwd_config)
except Exception:
    config_cwd = OmegaConf.create()
conf = OmegaConf.merge(config_base, config_cwd)


class Path:
    module = module_path
    repo = repo_path
    data = repo_path / "data"
    mzi = data / "mzi"
    mzi1 = mzi / "ZiheGao_MZI1_272_Scan1.mat"
    mzi11 = mzi / "ZiheGao_MZI1_273_Scan1.mat"
    mzi3 = mzi / "ZiheGao_MZI2_271_Scan1.mat"
    mzi1 = mzi / "ZiheGao_MZI3_270_Scan1.mat"
    mzi4 = mzi / "ZiheGao_MZI4_269_Scan1.mat"
    mzi5 = mzi / "ZiheGao_MZI5_268_Scan1.mat"
    mzi6 = mzi / "ZiheGao_MZI6_267_Scan1.mat"
    mzi8 = mzi / "ZiheGao_MZI8_266_Scan1.mat"
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


path = Path()

dbrs = [
    "ELEC_413_lukasc_BraggSet1Num10_1272.mat",
    "ELEC_413_lukasc_BraggSet1Num11_1273.mat",
    "ELEC_413_lukasc_BraggSet1Num12_1271.mat",
    "ELEC_413_lukasc_BraggSet1Num13_1278.mat",
    "ELEC_413_lukasc_BraggSet1Num14_1276.mat",
    "ELEC_413_lukasc_BraggSet1Num15_1277.mat",
    "ELEC_413_lukasc_BraggSet1Num16_1275.mat",
    "ELEC_413_lukasc_BraggSet1Num17_1282.mat",
    "ELEC_413_lukasc_BraggSet1Num18_1280.mat",
    "ELEC_413_lukasc_BraggSet1Num19_1281.mat",
    "ELEC_413_lukasc_BraggSet1Num1_1266.mat ",
    "ELEC_413_lukasc_BraggSet1Num20_1279.mat",
    "ELEC_413_lukasc_BraggSet1Num21_1286.mat",
    "ELEC_413_lukasc_BraggSet1Num22_1284.mat",
    "ELEC_413_lukasc_BraggSet1Num23_1285.mat",
    "ELEC_413_lukasc_BraggSet1Num24_1283.mat",
    "ELEC_413_lukasc_BraggSet1Num2_1264.mat ",
    "ELEC_413_lukasc_BraggSet1Num3_1265.mat ",
    "ELEC_413_lukasc_BraggSet1Num4_1263.mat ",
    "ELEC_413_lukasc_BraggSet1Num5_1270.mat ",
    "ELEC_413_lukasc_BraggSet1Num6_1268.mat ",
    "ELEC_413_lukasc_BraggSet1Num7_1269.mat ",
    "ELEC_413_lukasc_BraggSet1Num8_1267.mat ",
    "ELEC_413_lukasc_BraggSet1Num9_1274.mat ",
    "ELEC_413_lukasc_BraggSet2Num10_1248.mat",
    "ELEC_413_lukasc_BraggSet2Num11_1249.mat",
    "ELEC_413_lukasc_BraggSet2Num12_1247.mat",
    "ELEC_413_lukasc_BraggSet2Num13_1254.mat",
    "ELEC_413_lukasc_BraggSet2Num14_1252.mat",
    "ELEC_413_lukasc_BraggSet2Num15_1253.mat",
    "ELEC_413_lukasc_BraggSet2Num16_1251.mat",
    "ELEC_413_lukasc_BraggSet2Num17_1258.mat",
    "ELEC_413_lukasc_BraggSet2Num18_1256.mat",
    "ELEC_413_lukasc_BraggSet2Num19_1257.mat",
    "ELEC_413_lukasc_BraggSet2Num1_1242.mat ",
    "ELEC_413_lukasc_BraggSet2Num20_1255.mat",
    "ELEC_413_lukasc_BraggSet2Num21_1262.mat",
    "ELEC_413_lukasc_BraggSet2Num22_1260.mat",
    "ELEC_413_lukasc_BraggSet2Num23_1261.mat",
    "ELEC_413_lukasc_BraggSet2Num24_1259.mat",
    "ELEC_413_lukasc_BraggSet2Num2_1240.mat ",
    "ELEC_413_lukasc_BraggSet2Num3_1241.mat ",
    "ELEC_413_lukasc_BraggSet2Num4_1239.mat ",
    "ELEC_413_lukasc_BraggSet2Num5_1246.mat ",
    "ELEC_413_lukasc_BraggSet2Num6_1244.mat ",
    "ELEC_413_lukasc_BraggSet2Num7_1245.mat ",
    "ELEC_413_lukasc_BraggSet2Num8_1243.mat ",
    "ELEC_413_lukasc_BraggSet2Num9_1250.mat ",
    "ELEC_413_lukasc_BraggSet4Num10_1200.mat",
    "ELEC_413_lukasc_BraggSet4Num11_1201.mat",
    "ELEC_413_lukasc_BraggSet4Num12_1199.mat",
    "ELEC_413_lukasc_BraggSet4Num13_1206.mat",
    "ELEC_413_lukasc_BraggSet4Num14_1204.mat",
    "ELEC_413_lukasc_BraggSet4Num15_1205.mat",
    "ELEC_413_lukasc_BraggSet4Num16_1203.mat",
    "ELEC_413_lukasc_BraggSet4Num17_1210.mat",
    "ELEC_413_lukasc_BraggSet4Num18_1208.mat",
    "ELEC_413_lukasc_BraggSet4Num19_1209.mat",
    "ELEC_413_lukasc_BraggSet4Num1_1194.mat ",
    "ELEC_413_lukasc_BraggSet4Num20_1207.mat",
    "ELEC_413_lukasc_BraggSet4Num21_1214.mat",
    "ELEC_413_lukasc_BraggSet4Num22_1212.mat",
    "ELEC_413_lukasc_BraggSet4Num23_1213.mat",
    "ELEC_413_lukasc_BraggSet4Num24_1211.mat",
    "ELEC_413_lukasc_BraggSet4Num2_1192.mat ",
    "ELEC_413_lukasc_BraggSet4Num3_1193.mat ",
    "ELEC_413_lukasc_BraggSet4Num4_1191.mat ",
    "ELEC_413_lukasc_BraggSet4Num5_1198.mat ",
    "ELEC_413_lukasc_BraggSet4Num6_1196.mat ",
    "ELEC_413_lukasc_BraggSet4Num7_1197.mat ",
    "ELEC_413_lukasc_BraggSet4Num8_1195.mat ",
    "ELEC_413_lukasc_BraggSet4Num9_1202.mat ",
]


if __name__ == "__main__":
    print(path.data)
