import inspect

import ubcpdk
from ubcpdk.config import PATH

filepath = PATH.repo / "docs" / "components.rst"

skip = {
    "LIBRARY",
    "circuit_names",
    "cells",
    "component_names",
    "container_names",
    "component_names_test_ports",
    "component_names_skip_test",
    "component_names_skip_test_ports",
    "dataclasses",
    "library",
    "waveguide_template",
    "add_ports_m1",
    "add_ports_m2",
    "add_ports",
    "import_gds",
}

skip_plot = {}
skip_settings = {"flatten", "safe_cell_names"}


with open(filepath, "w+") as f:
    f.write(
        """

Cells summary
=============================

.. currentmodule:: ubcpdk.components

.. autosummary::
   :toctree: _autosummary/

"""
    )

    for name in sorted(ubcpdk.cells.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(ubcpdk.cells[name])
        kwargs = ", ".join(
            [
                f"{p}={repr(sig.parameters[p].default)}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, int | float | str | tuple)
                and p not in skip_settings
            ]
        )
        f.write(f"   {name}\n")
