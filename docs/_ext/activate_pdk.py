"""Sphinx extension: activate ubcpdk PDK for docs build.

Autodoc renders cell docstrings that instantiate partials requiring an
active PDK. The package itself no longer activates on import, so we
activate here for the docs build only.
"""

import ubcpdk


def setup(app):
    ubcpdk.PDK.activate()
    return {"version": "0.1", "parallel_read_safe": True}
