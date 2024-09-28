from __future__ import annotations

import gdsfactory as gf
import jsondiff
import pytest
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
    "add_pads_rf",
    "dbr",
    "dbg",
    "import_gds",
    "import_gc",
    "pad_array",
    "add_pads_dc",
    "ebeam_adiabatic_te1550",
    "ebeam_adiabatic_tm1550",
    "ebeam_splitter_adiabatic_swg_te1550",
    "ebeam_swg_edgecoupler",
    "ebeam_BondPad",
    "add_fiber_array",
    "add_pads_top",
    "add_pads_bot",
    "wire_corner",
    "straight_heater_metal",
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
    allow_multiple = True
    n = c.get_netlist(allow_multiple=allow_multiple)
    n.pop("connections", None)
    n.pop("warnings", None)
    if check:
        data_regression.check(n)

    yaml_str = c.write_netlist(n)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist(allow_multiple=allow_multiple)

    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("ports", None)
    assert len(d) == 0, d


if __name__ == "__main__":
    component_type = "straight_heater_metal"
    component_type = "gc_te1310_broadband"
    component_type = "ring_double"
    component_type = "terminator_short"
    component_type = "mzi_heater"
    component_type = "ring_double_heater"
    connection_error_types = {
        "optical": ["width_mismatch", "shear_angle_mismatch", "orientation_mismatch"]
    }
    connection_error_types = {"optical": []}

    c1 = cells[component_type]()
    # c1.show()
    n = c1.get_netlist(
        allow_multiple=True, connection_error_types=connection_error_types
    )
    yaml_str = c1.write_netlist(n)
    c1.delete()
    # gf.clear_cache()
    # print(yaml_str)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist(allow_multiple=True)
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    c2.show()
    assert len(d) == 0, d
