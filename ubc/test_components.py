import pytest
from pp.component import Component
from pp.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture
from ubc import component_factory, component_names, container_factory
from ubc.add_gc import add_gc
from ubc.waveguide import waveguide


@pytest.fixture(params=component_names, scope="function")
def component(request) -> Component:
    return component_factory[request.param]()


def test_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    difftest(component)


def test_settings(component: Component, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.get_settings())


def test_ports(component: Component, num_regression: NumericRegressionFixture) -> None:
    """Avoid regressions in port names and locations."""
    if component.ports:
        num_regression.check(component.get_ports_array())
