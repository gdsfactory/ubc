# Bragg Gratings

Design objective:

- Determine the grating strength coupling parameter (kagdsfactory., ∆n, or bandwidth) versus the corrugation width (∆w), and compare rectangular versus sinusoidal gratings.
- Compare experimental data with the simulation data (3D-FDTD Bloch agdsfactory.oach).
- Update the CML model based on this data.

Design:

- Measure the reflectivity spectrum (only).  This only requires two grating couplers.  Removing the 3rd grating coupler that was previously used to measure the transmission eliminates a source of back-reflection (-20 dB).  Instead of the 3rd grating coupler, we use a terminator (-30 dB back reflection).
- Make the layout very compact, particularly the gratings themselves, so that we minimize the manufacturing variability. Namely, we want the central wavelength to be as similar as possible.
- For each layout block, we have 24 gratings, connected to 24 pairs of grating couplers.  The grating couplers are interlaced to minimize space.
- Check for a wide range of ∆W to look for saturation and nonlinearity (as shown by James Pond’s simulation).  Also check a narrow range to look for mask quantization errors.
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

# Set 1

Design # | ∆w | Type
1 | 5 | rectangular
2 | 200 | sinusoidal
3 | 10 | rectangular
4 | 180 | sinusoidal
5 | 20 | rectangular
6 | 160 | sinusoidal
7 | 40 | rectangular
8 | 140 | sinusoidal
9 | 60 | rectangular
10 | 120 | sinusoidal
11 | 80 | rectangular
12 | 100 | sinusoidal
13 | 100 | rectangular
14 | 80 | sinusoidal
15 | 120 | rectangular
16 | 60 | sinusoidal
17 | 140 | rectangular
18 | 40 | sinusoidal
19 | 160 | rectangular
20 | 20 | sinusoidal
21 | 180 | rectangular
22 | 10 | sinusoidal
23 | 200 | rectangular
24 | 5 | sinusoidal

Set 4:

Design #| ∆w| Type
1 | 20 | rectangular
2 | 31 | sinusoidal
3 | 21 | rectangular
4 | 30 | sinusoidal
5 | 22 | rectangular
6 | 20 | sinusoidal
7 | 23 | rectangular
8 | 28 | sinusoidal
9 | 24 | rectangular
10 | 27 | sinusoidal
11 | 25 | rectangular
12 | 26 | sinusoidal
13 | 26 | rectangular
14 | 25 | sinusoidal
15 | 27 | rectangular
16 | 24 | sinusoidal
17 | 28 | rectangular
18 | 23 | sinusoidal
19 | 29 | rectangular
20 | 22 | sinusoidal
21 | 30 | rectangular
22 | 21 | sinusoidal
23 | 31 | rectangular
24 | 20 | sinusoidal
