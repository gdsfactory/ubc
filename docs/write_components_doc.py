import pathlib
import inspect
import ubc


filepath = pathlib.Path(__file__).parent.absolute() / "components.rst"

skip = {
    "LIBRARY",
    "circuit_names",
    "component_factory",
    "component_names",
    "container_names",
    "component_names_test_ports",
    "component_names_skip_test",
    "component_names_skip_test_ports",
    "dataclasses",
    "library",
    "waveguide_template",
}

skip_plot = {}
skip_settings = {}


with open(filepath, "w+") as f:
    f.write(
        """

Here is a list of generic component factories that you can customize for your fab or use it as an inspiration to build your own.


Components
=============================
"""
    )

    for name in sorted(ubc.components.factory.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(ubc.components.factory[name])
        kwargs = ", ".join(
            [
                f"{p}={repr(sig.parameters[p].default)}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, (int, float, str, tuple))
                and p not in skip_settings
            ]
        )
        if name in skip_plot:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: ubc.components.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: ubc.components.{name}

.. plot::
  :include-source:

  import ubc

  c = ubc.components.{name}({kwargs})
  c.plot()

"""
            )
