from __future__ import annotations

import gdsfactory as gf
import jsondiff
import pytest
from omegaconf import OmegaConf
from pytest_regressions.data_regression import DataRegressionFixture

from ubcpdk import PDK

cells = PDK.cells
skip_test = {
    "via_stack_heater_mtop",
    "add_pins_bbox_siepic_metal",
    "add_pins_bbox_siepic",
    "add_pins_siepic",
    "add_pins_siepic_metal",
    "add_pads",
    "dbr",
    "dbg",
    "import_gds",
    "import_gc",
    "pad_array",
    "add_pads_dc",
    "ebeam_adiabatic_te1550",
}
cell_names = cells.keys() - skip_test


@pytest.mark.parametrize("component_type", cell_names)
def test_netlists(
    component_type: str,
    data_regression: DataRegressionFixture,
    check: bool = True,
    component_factory=cells,
) -> None:
    """Write netlists for hierarchical circuits.
    Checks that both netlists are the same jsondiff does a hierarchical diff.
    Component -> netlist -> Component -> netlist

    Args:
        component_type: component type.
        data_regression: regression testing fixture.
        check: whether to check the netlist.
        component_factory: component factory.
    """
    c = component_factory[component_type]()
    n = c.get_netlist()
    if check:
        data_regression.check(n)

    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = gf.read.from_yaml(yaml_str, name=c.name)
    n2 = c2.get_netlist()

    d = jsondiff.diff(n, n2)
    assert len(d) == 0, d
