import pytest
from pp.component import Component
from pp.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture
from ubc.pdk import COMPONENT_FACTORY


component_factory = COMPONENT_FACTORY.factory
component_names = component_factory.keys()


@pytest.fixture(params=component_names, scope="function")
def component(request) -> Component:
    return component_factory[request.param](cache=False)


def test_pdk_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_pdk_settings(
    component: Component, data_regression: DataRegressionFixture
) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.get_settings())


def test_pdk_ports(
    component: Component, num_regression: NumericRegressionFixture
) -> None:
    """Avoid regressions in port names and locations."""
    if component.ports:
        num_regression.check(component.get_ports_array())


def test_assert_ports_on_grid(component: Component):
    component.assert_ports_on_grid()
