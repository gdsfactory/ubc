""" [Optional] Bragg Gratings
Design objective:

- Determine the grating strength coupling parameter (kappa, ∆n, or bandwidth) versus the corrugation width (∆w), and compare rectangular versus sinusoidal gratings.
- Compare experimental data with the simulation data (3D-FDTD Bloch approach).
- Update the CML model based on this data.

Design:

- Measure the reflectivity spectrum (only).  This only requires two grating couplers.  Removing the 3rd grating coupler that was previously used to measure the transmission eliminates a source of back-reflection (-20 dB).  Instead of the 3rd grating coupler, we use a terminator (-30 dB back reflection).
- Make the layout very compact, particularly the gratings themselves, so that we minimize the manufacturing variability. Namely, we want the central wavelength to be as similar as possible.
- For each layout block, we have 24 gratings, connected to 24 pairs of grating couplers.  The grating couplers are interlaced to minimize space.
- Check for a wide range of ∆W to look for saturation and nonlinearity (as shown by James Pond’s simulations).  Also check a narrow range to look for mask quantization errors.
- Parameter range, set 1-2:
-     ∆W = [5, 10, 20:20:200 nm] (total 12)
-     type = [rectangular, sinusoidal]
-     N = [200, 400] (number of gratings)
- Parameter range, set 4:
-     ∆W = [20:1:31] (total 12)
-     type = [rectangular, sinusoidal]
-     N = [200] (number of gratings)

The three layouts are as follows:

- ELEC413_lukasc_A.gds
-     N = 200, set 1
- ELEC413_lukasc_B.gds
-     N = 400, set 2
- ELEC413_lukasc_D.gds
-     N = 200, set 4

Each contains 24 gratings, connected in reflection mode using a 2x2 splitter.
Sets 1, 2:


Design # | ∆w  | Type
-------- | --  | ---------
1        | 5   | rectangular
2        | 200 | sinusoidal
3        | 10  | rectangular
4        | 180 | sinusoidal
5        | 20  | rectangular
6        | 160 | sinusoidal
7        | 40  | rectangular
8        | 140 | sinusoidal
9        | 60  | rectangular
10       | 120 | sinusoidal
11       | 80  | rectangular
12       | 100 | sinusoidal
13       | 100 | rectangular
14       | 80  | sinusoidal
15       | 120 | rectangular
16       | 60  | sinusoidal
17       | 140 | rectangular
18       | 40  | sinusoidal
19       | 160 | rectangular
20       | 20  | sinusoidal
21       | 180 | rectangular
22       | 10  | sinusoidal
23       | 200 | rectangular
24       | 5   | sinusoidal

Set 4:

Design #| ∆w | Type
------- | --  | ---------
1       | 20 | rectangular
2       | 31 | sinusoidal
3       | 21 | rectangular
4       | 30 | sinusoidal
5       | 22 | rectangular
6       | 20 | sinusoidal
7       | 23 | rectangular
8       | 28 | sinusoidal
9       | 24 | rectangular
10      | 27 | sinusoidal
11      | 25 | rectangular
12      | 26 | sinusoidal
13      | 26 | rectangular
14      | 25 | sinusoidal
15      | 27 | rectangular
16      | 24 | sinusoidal
17      | 28 | rectangular
18      | 23 | sinusoidal
19      | 29 | rectangular
20      | 22 | sinusoidal
21      | 30 | rectangular
22      | 21 | sinusoidal
23      | 31 | rectangular
24      | 20 | sinusoidal

"""
from ubc.config import path

dbrs = {
    filename.split("_")[3][8:].replace("Num", "_"): path.dbr / filename
    for filename in [
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
        "ELEC_413_lukasc_BraggSet1Num1_1266.mat",
        "ELEC_413_lukasc_BraggSet1Num20_1279.mat",
        "ELEC_413_lukasc_BraggSet1Num21_1286.mat",
        "ELEC_413_lukasc_BraggSet1Num22_1284.mat",
        "ELEC_413_lukasc_BraggSet1Num23_1285.mat",
        "ELEC_413_lukasc_BraggSet1Num24_1283.mat",
        "ELEC_413_lukasc_BraggSet1Num2_1264.mat",
        "ELEC_413_lukasc_BraggSet1Num3_1265.mat",
        "ELEC_413_lukasc_BraggSet1Num4_1263.mat",
        "ELEC_413_lukasc_BraggSet1Num5_1270.mat",
        "ELEC_413_lukasc_BraggSet1Num6_1268.mat",
        "ELEC_413_lukasc_BraggSet1Num7_1269.mat",
        "ELEC_413_lukasc_BraggSet1Num8_1267.mat",
        "ELEC_413_lukasc_BraggSet1Num9_1274.mat",
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
        "ELEC_413_lukasc_BraggSet2Num1_1242.mat",
        "ELEC_413_lukasc_BraggSet2Num20_1255.mat",
        "ELEC_413_lukasc_BraggSet2Num21_1262.mat",
        "ELEC_413_lukasc_BraggSet2Num22_1260.mat",
        "ELEC_413_lukasc_BraggSet2Num23_1261.mat",
        "ELEC_413_lukasc_BraggSet2Num24_1259.mat",
        "ELEC_413_lukasc_BraggSet2Num2_1240.mat",
        "ELEC_413_lukasc_BraggSet2Num3_1241.mat",
        "ELEC_413_lukasc_BraggSet2Num4_1239.mat",
        "ELEC_413_lukasc_BraggSet2Num5_1246.mat",
        "ELEC_413_lukasc_BraggSet2Num6_1244.mat",
        "ELEC_413_lukasc_BraggSet2Num7_1245.mat",
        "ELEC_413_lukasc_BraggSet2Num8_1243.mat",
        "ELEC_413_lukasc_BraggSet2Num9_1250.mat",
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
        "ELEC_413_lukasc_BraggSet4Num1_1194.mat",
        "ELEC_413_lukasc_BraggSet4Num20_1207.mat",
        "ELEC_413_lukasc_BraggSet4Num21_1214.mat",
        "ELEC_413_lukasc_BraggSet4Num22_1212.mat",
        "ELEC_413_lukasc_BraggSet4Num23_1213.mat",
        "ELEC_413_lukasc_BraggSet4Num24_1211.mat",
        "ELEC_413_lukasc_BraggSet4Num2_1192.mat",
        "ELEC_413_lukasc_BraggSet4Num3_1193.mat",
        "ELEC_413_lukasc_BraggSet4Num4_1191.mat",
        "ELEC_413_lukasc_BraggSet4Num5_1198.mat",
        "ELEC_413_lukasc_BraggSet4Num6_1196.mat",
        "ELEC_413_lukasc_BraggSet4Num7_1197.mat",
        "ELEC_413_lukasc_BraggSet4Num8_1195.mat",
        "ELEC_413_lukasc_BraggSet4Num9_1202.mat",
    ]
}
