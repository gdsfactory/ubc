import pytest
from ubc import _components, component_type2factory
from ubc.add_gc import add_gc
from ubc.waveguide import waveguide

_containers = [
    add_gc,
]


@pytest.mark.parametrize("function", _containers)
def test_properties_containers(function, data_regression):
    component = waveguide()
    c = function(component=component)
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("component_type", _components)
def test_properties(component_type, data_regression):
    c = component_type2factory[component_type]()
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("component_type", _components)
def test_ports(component_type, num_regression):
    c = component_type2factory[component_type]()
    if c.ports:
        num_regression.check(c.get_ports_array())


if __name__ == "__main__":
    test_properties()
