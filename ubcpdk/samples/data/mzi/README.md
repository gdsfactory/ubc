Measuring optical waveguide length:
- measure the vertical path, from centre to centre  (path is the one with sharp angles)\
- determine radius (from PCell)\
- add the radial bend waveguide, subtract the path in that same place\
-
MZI1:\
dL_wg=0\
\
MZI2: \
r=5\
dL_path = (208.40000 - 148.15000) * 2\
dL_wg = dL_path + 2*pi*r - 4*2*r \
= 111.915\
\
MZI3:\
r=5\
dL_path = (259.55000-148.15000) * 2\
dL_wg = dL_path + 2*pi*r - 4*2*r ; dL_wg\
= 214.215\
\
MZI4:\
r1 = 435.90000-427.60000; r1\
r2 = 10\
dL_path = (259.55000-148.15000) * 2\
dL_wg = dL_path + pi*(r1+r2) - 4*(r1+r2) ; dL_wg\
= 207.08945\
\
MZI5: \
r1 = 556.35000-547.60000; r1\
r2 = 10\
dL_path = (259.55000-148.15000) * 2\
dL_wg = dL_path + pi*(r1+r2) - 4*(r1+r2) ; dL_wg\
= 206.703125\
\
MZI6:\
r=4\
dL_path = (259.55000-148.15000) * 2\
dL_wg = dL_path + 2*pi*r - 4*2*r ; dL_wg\
= 215.932\
\
MZI8:\
r=3\
dL_path = (259.55000-148.15000) * 2\
dL_wg = dL_path + 2*pi*r - 4*2*r ; dL_wg\
= 217.649\
\
MZI17:\
r=2\
dL_path = (259.55000-148.15000) * 2\
dL_wg = dL_path + 2*pi*r - 4*2*r ; dL_wg\


## Lukas

Measuring optical waveguide length:\
- measure the horizontal path, from centre to centre  (path is the one with sharp angles)\
- add 2 * ybranch output separation\
\
Can also measure the waveguide by finding the total area\
{\field{\*\fldinst{HYPERLINK "http://www.klayout.de/useful_scripts.html#calc_area.rbm"}}{\fldrslt http://www.klayout.de/useful_scripts.html#calc_area.rbm}}\
then divide by the waveguide width\
\
MZI1:\
dL_wg=(85.10000-77.20000)*2  +5.5*2; dL_wg\
= 26.8\
\
or, \
area1 = 80.210502\
area2 = 66.810502\
dL_wg = (80.210502-66.810502)/0.5; dL_wg\
= 26.8\
\
MZI2:\
dL_wg=(768.95000-77.20000)*2 +5.5*2; dL_wg\
= 1394.5\
