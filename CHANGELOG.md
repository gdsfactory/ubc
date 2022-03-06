# Changelog

## [1.3.5]

## [1.3.4](https://github.com/gdsfactory/ubc/pull/8)

- add pins also adds the device recognition layer

## 1.3.3

- enable dispersive fdtd simulations in tidy3d thanks to gdsfactory 4.3.4

## [1.3.1](https://github.com/gdsfactory/ubc/pull/7)

- update gdsfactory>=4.2.16
- add tidy3d simulations

## 1.3.0

- update gdsfactory>=4.2.1

## 1.1.0

- remove triangle from requirements.txt
- enforce gdsfactory>=4.0.0
- simplify grating couplers definition
- add meep and lumerical simulation functions

## 1.0.6

- move ubcpdk.simulations to ubcpdk.simulation for consistency with gdsfactory
- add ubcpdk.simulation.gmeep
- move tests to ubcpdk/tests

## 1.0.5

- move sample data into the module

## 1.0.0

- rename package from ubc to ubcpdk to match pypi name
- move ubcsp into ubcpdk/simulation/circuits
- rename ubcpdk/da as ubcpdk/data

## 0.0.12

- fix installation by addding lyp files to MANIFEST
- compatible with latest gdsfactory
- `pip install ubcpdk` has less dependencies than `pip install ubc[full]`

## 0.0.6

- compatible with gdsfactory 3.9.12
- merge mask metadata
- replace `-` with `_` in measurement names

## 0.0.4

- compatible with gdsfactory 3.1.5

## 0.0.3

- components in different folders

## 0.0.2

- added pins to components
- added notebooks
