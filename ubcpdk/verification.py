from pathlib import Path

import gdsfactory
import klayout.db as kdb
import pya
import SiEPIC
import SiEPIC.verification
from SiEPIC.utils import get_technology_by_name, klive

from ubcpdk.config import PATH


def layout_check(
    component: gdsfactory.Component,
    klayout_tech_path: str | Path | None = PATH.lyt,
    show_klive: bool = False,
) -> int:
    """Run a layout check using SiEPIC-Tool's functional verification.

    Args:
        component: gdsfactory component.
        klayout_tech_path: path to the klayout technology folder.
        show_klive: show results in KLayout.
    """

    gdspath = component.write_gds()

    # load in KLayout database
    ly = pya.Layout()

    # load SiEPIC technology
    ly.TECHNOLOGY = get_technology_by_name("UBCPDK")

    ly.read(str(gdspath))
    if len(ly.top_cells()) != 1:
        raise ValueError("Layout can only have one top cell")
    topcell = ly.top_cell()

    tech = kdb.Technology()
    tech.load(str(klayout_tech_path))

    # perform verification
    file_lyrdb = str(gdspath) + ".lyrdb"
    num_errors = SiEPIC.verification.layout_check(cell=topcell, file_rdb=file_lyrdb)

    if show_klive:
        klive.show(gdspath, lyrdb_filename=file_lyrdb)

    return num_errors


if __name__ == "__main__":
    import gdsfactory as gf

    file_path = PATH.repo / "tests" / "tests" / "mmi2x2.oas"
    c = gf.import_gds(file_path, read_metadata=True)

    layout_check(c)
