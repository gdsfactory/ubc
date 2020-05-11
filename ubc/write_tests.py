"""
write regression tests for component ports and properties
"""

from ubc import component_type2factory

_skip_test = set()
_skip_test_ports = set()


def write_test_properties():
    """ writes a regression test for all the component properties dict"""
    with open("test_components.py", "w") as f:
        f.write(
            "# this code has been automatically generated from ubc/write_tests.py\n"
        )
        f.write("import ubc\n\n")

        for c in set(component_type2factory.keys()) - _skip_test:
            f.write(
                f"""
def test_{c}(data_regression):
    c = ubc.{c}()
    data_regression.check(c.get_settings())

"""
            )


def write_test_ports():
    """ writes a regression test for all the ports """
    with open("test_ports.py", "w") as f:
        f.write(
            "# this code has been automatically generated from ubc/write_tests.py\n"
        )
        f.write("import ubc\n\n")

        for component_function in (
            set(component_type2factory.values()) - _skip_test_ports
        ):
            c = component_function.__name__
            if component_function().ports:
                f.write(
                    f"""
def test_{c}(num_regression):
    c = ubc.{c}()
    num_regression.check(c.get_ports_array())

    """
                )


if __name__ == "__main__":
    write_test_properties()
    write_test_ports()
