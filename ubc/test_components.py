import pytest
from pp.difftest import difftest
from ubc import component_factory, component_names, container_factory
from ubc.add_gc import add_gc
from ubc.waveguide import waveguide


@pytest.mark.parametrize("component_type", component_names)
def test_settings(component_type, data_regression):
    """Avoid regressions when exporting settings."""
    c = component_factory[component_type]()
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("component_type", component_names)
def test_ports(component_type, num_regression):
    """Avoid regressions in port names and locations."""
    c = component_factory[component_type]()
    if c.ports:
        num_regression.check(c.get_ports_array())


@pytest.mark.parametrize("component_type", component_names)
def test_gds(component_type):
    """Avoid regressions in GDS geometry shapes and layers."""
    c = component_factory[component_type]()
    difftest(c)
