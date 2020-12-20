import pytest
from lytest import contained_phidlDevice, difftest_it
from ubc import _components, component_factory
from ubc.add_gc import add_gc
from ubc.waveguide import waveguide

# @pytest.mark.parametrize("component_type", _components)
# def test_gds(component_type):
#     @contained_phidlDevice
#     def device(top):
#         c = component_factory[component_type]()
#         top.add_ref(c)

#     def test_gds():
#         difftest_it(device)()


# if __name__ == "__main__":
#     test_gds(component_type="ring")
