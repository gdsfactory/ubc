import pytest
from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture
from ubcpdk.components import factory


component_names = factory.keys()


@pytest.fixture(params=component_names, scope="function")
def component(request) -> Component:
    return factory[request.param]()


def test_pdk_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_pdk_settings(
    component: Component, data_regression: DataRegressionFixture
) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component: Component):
    component.assert_ports_on_grid()
