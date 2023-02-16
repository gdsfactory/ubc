# Changelog

## [1.20.0](https://github.com/gdsfactory/ubc/pull/194)

- update floorplan size
- fix optical labels, make sure optical and electrical labels match
- fix pads spacing to 125um
- flip orientation of heated rings

## [1.19.0](https://github.com/gdsfactory/ubc/pull/186)

- update to gdsfactory 6.37.0
- fix drc errors

## [1.18.0](https://github.com/gdsfactory/ubc/pull/158)

- update to gdsfactory 6.35.1
- use bbox_layers

## [1.15.0](https://github.com/gdsfactory/ubc/pull/158)

- update to gdsfactory 6.21.1
- use o1, o2, o3 port naming convention instead of opt1, opt2, opt3

## [1.14.0](https://github.com/gdsfactory/ubc/pull/157)
- update to gdsfactory 6.20.0

## [1.13.0](https://github.com/gdsfactory/ubc/pull/146)

- update to gdsfactory 6.19.0
- simpler installer

## 1.12.0

- update to gdsfactory 6.16.3

## 1.11.0

- update to gdsfactory 6.16.1 [PR](https://github.com/gdsfactory/ubc/pull/146)

## 1.10.0

- update to gdsfactory 6.2.1 [PR](https://github.com/gdsfactory/ubc/pull/120)

## [1.9.0](https://github.com/gdsfactory/ubc/pull/116)

- update to gdsfactory 6.0.3


## 1.8.0

- update to gdsfactory 5.50.3

## [1.6.6](https://github.com/gdsfactory/ubc/pull/65)

- update to gdsfactory 5.13.0


## [1.6.5](https://github.com/gdsfactory/ubc/pull/57)

- update to gdsfactory 5.12.9

## [1.6.4](https://github.com/gdsfactory/ubc/pull/52)

- update interconnect plugin [PR](https://github.com/gdsfactory/ubc/pull/51)

## [1.6.1](https://github.com/gdsfactory/ubc/pull/49)

- update to gdsfactory 5.10.1

## [1.6.0](https://github.com/gdsfactory/ubc/pull/45)

- update to gdsfactory 5.8.7

## [1.5.10](https://github.com/gdsfactory/ubc/pull/43)

- update to gdsfactory 5.7.1

## [1.5.9](https://github.com/gdsfactory/ubc/pull/41)

- update to latest gdsfactory

## [1.5.8](https://github.com/gdsfactory/ubc/pull/39)

- update to latest gdsfactory

## [1.5.7](https://github.com/gdsfactory/ubc/pull/31)

- update to gdsfactory 5.5.1

## [1.5.6](https://github.com/gdsfactory/ubc/pull/30)

- update to gdsfactory 5.4.2

## [1.5.5](https://github.com/gdsfactory/ubc/pull/29)

- update gdsfactory
- register `*.pic.yml`

## [1.5.4](https://github.com/gdsfactory/ubc/pull/27)

- update gdsfactory

## [1.5.3](https://github.com/gdsfactory/ubc/pull/26)

- [PR](https://github.com/gdsfactory/ubc/pull/24) replaced bbox with Section in strip cross_section
- [PR](https://github.com/gdsfactory/ubc/pull/25) assumes that the imported gds file is already siepic-compatible and adds ports where it finds siepic pins.

## [1.5.2](https://github.com/gdsfactory/ubc/pull/23)

- compatible with gdsfactory 5.2.0

## [1.5.1](https://github.com/gdsfactory/ubc/pull/22)

- add fiber accepts ComponentSpec
- register gdsfactory containers together with cells

## [1.5.0](https://github.com/gdsfactory/ubc/pull/21)

- update tests to pass for gdsfactory 5.0.1

## [1.4.2](https://github.com/gdsfactory/ubc/pull/20)

- rename component_factory to cells and cross_section_factory to cross_sections


## [1.4.1](https://github.com/gdsfactory/ubc/pull/19)

- update layer_stack to be compatible with latest gdsfactory 4.7.1

## [1.4.0](https://github.com/gdsfactory/ubc/pull/18)

- simpler component_factory thanks to `import_module_factories` function from gf.cell

## [1.3.9](https://github.com/gdsfactory/ubc/pull/17)

- add circuit samples

## [1.3.7](https://github.com/gdsfactory/ubc/pull/15)

- add interconnect [plugin](https://github.com/gdsfactory/ubc/pull/14)

## [1.3.6](https://github.com/gdsfactory/ubc/pull/11)

- change pin length from 100nm to 10nm

## [1.3.5](https://github.com/gdsfactory/ubc/pull/9)

- pins are compatible with siepic

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

- fix installation by adding lyp files to MANIFEST
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
