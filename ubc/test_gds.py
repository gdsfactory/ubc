import pytest
from pp.testing import difftest
from ubc import _components, component_factory
from ubc.add_gc import add_gc
from ubc.waveguide import waveguide

container_factory = dict(add_gc=add_gc)


@pytest.mark.parametrize("component_type", _components)
def test_gds_components(component_type):
    component = component_factory[component_type]()
    difftest(component)


@pytest.mark.parametrize("component_type", container_factory.keys())
def test_gds_containers(component_type):
    component = container_factory[component_type](component=waveguide())
    difftest(component)


if __name__ == "__main__":
    test_gds_components(component_type="ring")
