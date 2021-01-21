import pytest
from pp.testing import difftest
from ubc import container_factory
from ubc.add_gc import add_gc
from ubc.waveguide import waveguide

container_names = container_factory.keys()
component = waveguide()


@pytest.mark.parametrize("container_type", container_names)
def test_settings(container_type, data_regression):
    """Avoid regressions when exporting settings."""
    c = container_factory[container_type](component=component)
    data_regression.check(c.get_settings())


@pytest.mark.parametrize("container_type", container_names)
def test_ports(container_type, num_regression):
    """Avoid regressions in port names and locations."""
    c = container_factory[container_type](component=component)
    if c.ports:
        num_regression.check(c.get_ports_array())


@pytest.mark.parametrize("container_type", container_names)
def test_gds(container_type):
    """Avoid regressions in GDS geometry shapes and layers."""
    c = container_factory[container_type](component=component)
    difftest(c)


if __name__ == "__main__":
    test_gds(container_type="add_gc")
