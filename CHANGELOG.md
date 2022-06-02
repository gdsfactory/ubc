# Changelog

## [1.6.0]

- update to gdsfactory 5.8.6

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
