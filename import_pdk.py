"""From a list of GDS files, generate a script to import the cells from a pdk."""

import pathlib

from gdsfactory import logger
from gdsfactory.typings import PathType
from gdsfactory.write_cells import clean_name

from ubcpdk.config import PATH

script_prefix = """
from pathlib import Path
from functools import partial
import gdsfactory as gf

from ubcpdk.config import PATH
from ubcpdk.import_gds import import_gds

"""

prefix_ant = """
gdsdir = PATH.gds_ant
"""

prefix_ebeam = """
gdsdir = PATH.gds_ebeam
"""

prefix_beta = """
gdsdir = PATH.gds_beta
"""
prefix_dream = """
gdsdir = PATH.gds_dream
"""
prefix_single = """
gdsdir = PATH.gds_single
"""

script_prefix_ant = script_prefix + prefix_ant
script_prefix_ebeam = script_prefix + prefix_ebeam
script_prefix_beta = script_prefix + prefix_beta
script_prefix_dream = script_prefix + prefix_dream
script_prefix_single = script_prefix + prefix_single


def get_script(
    gdspath: PathType, module: str | None = None, skip: list[str] | None = None
) -> str:
    """Returns script for importing a fixed cell.

    Args:
        gdspath: fixed cell gdspath.
        module: if any includes plot directive.
        skip: list of cells to skip.

    """
    skip = skip or []
    gdspath = pathlib.Path(gdspath)
    cell = clean_name(gdspath.stem)

    for s in skip:
        if s in cell:
            return ""

    gdspath = gdspath.stem + gdspath.suffix

    package = module.split(".")[0] if module and "." in module else module
    if module:
        return f"""

@gf.cell
def {cell}()->gf.Component:
    '''Returns {cell} fixed cell.

    .. plot::
      :include-source:

      import {package}

      c = {module}.{cell}()
      c.plot()
    '''
    return import_gds(gdsdir/{str(gdspath)!r})

"""

    return f"""

@gf.cell
def {cell}()->gf.Component:
    '''Returns {cell} fixed cell.'''
    return import_gds(gdsdir/{str(gdspath)!r})

"""


def get_import_gds_script(
    dirpath: PathType,
    module: str | None = None,
    skip: list[str] | None = None,
    script_prefix=script_prefix_single,
) -> str:
    """Returns import_gds script from a directory with all the GDS files.

    Args:
        dirpath: fixed cell directory path.
        module: Optional plot directive to plot imported component.
        skip: list of strings to skip.
        script_prefix: prefix for the script.

    """
    dirpath = pathlib.Path(dirpath)
    if not dirpath.exists():
        raise ValueError(f"{str(dirpath.absolute())!r} does not exist.")

    gdspaths = list(dirpath.glob("*.gds")) + list(dirpath.glob("*.GDS"))

    if not gdspaths:
        raise ValueError(f"No GDS files found at {dirpath.absolute()!r}.")

    logger.info(f"Writing {len(gdspaths)} cells from {dirpath.absolute()!r}")

    script = [script_prefix]
    cells = [get_script(gdspath, module=module, skip=skip) for gdspath in gdspaths]
    script += sorted(cells)
    return "\n".join(script)


if __name__ == "__main__":
    s = get_import_gds_script(PATH.gds_ant, skip=[], script_prefix=script_prefix_ant)
    PATH.fixed_ant.write_text(s)

    s = get_import_gds_script(
        PATH.gds_ebeam, skip=[], script_prefix=script_prefix_ebeam
    )
    PATH.fixed_ebeam.write_text(s)

    s = get_import_gds_script(PATH.gds_beta, skip=[], script_prefix=script_prefix_beta)
    PATH.fixed_beta.write_text(s)

    s = get_import_gds_script(
        PATH.gds_dream, skip=[], script_prefix=script_prefix_dream
    )
    PATH.fixed_dream.write_text(s)

    s = get_import_gds_script(
        PATH.gds_single, skip=[], script_prefix=script_prefix_single
    )
    PATH.fixed_single.write_text(s)
