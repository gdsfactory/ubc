import pathlib
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from ubcpdk import cells


skip_test = {
    "add_siepic_labels",
    "add_fiber_array_siepic",
    "add_pins_bbox_siepic_metal",
    "add_pins_siepic",
    "add_pins_siepic_metal",
    "dbr",
    "add_pins_bbox_siepic",
    "add_pads",
    "add_pins_bbox_siepic_remove_layers",
}
cell_names = set(cells.keys()) - set(skip_test)
dirpath_ref = pathlib.Path(__file__).absolute().parent / "ref"


@pytest.fixture(params=cell_names, scope="function")
def component(request) -> Component:
    return cells[request.param]()


def test_pdk_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry, cell names and layers."""
    difftest(component, dirpath=dirpath_ref)


def test_pdk_settings(
    component: Component, data_regression: DataRegressionFixture
) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component: Component):
    component.assert_ports_on_grid()
