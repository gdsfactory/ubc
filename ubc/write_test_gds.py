from ubc import _components, component_factory
from ubc.add_gc import add_gc

container_factory = dict(add_gc=add_gc)


imports = """
import ubc
from ubc.waveguide import waveguide as waveguide_function
from ubc.write_test_gds import container_factory
from lytest import contained_phidlDevice, difftest_it
"""

if __name__ == "__main__":
    with open("test_gds.py", "w") as f:
        f.write(imports)
        for component_type in sorted(list(_components)):
            f.write(
                f"""

@contained_phidlDevice
def {component_type}(top):
    top.add_ref(ubc.{component_type}())


def test_gds_{component_type}():
    difftest_it({component_type})()
"""
            )
        for container_type in container_factory.keys():
            f.write(
                f"""

@contained_phidlDevice
def {container_type}(top):
    component = waveguide_function()
    container_function = container_factory["{container_type}"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_{container_type}():
    difftest_it({container_type})()
"""
            )
