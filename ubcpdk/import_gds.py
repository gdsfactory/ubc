from collections.abc import Callable
from functools import cache
from pathlib import Path

import gdsfactory as gf
import kfactory as kf
from gdsfactory.component import Component
from gdsfactory.read.import_gds import kcell_to_component
from kfactory import KCLayout

from ubcpdk.tech import LAYER


@cache
def import_gds(
    gdspath: str | Path,
    cellname: str | None = None,
    port_type="optical",
    layer_pin=LAYER.PORT,
    layer_port=LAYER.WG,
    post_process: Callable[[Component], Component] | None = None,
) -> Component:
    """Returns klayout cell from GDS."""
    temp_kcl = KCLayout(name=str(gdspath))
    temp_kcl.read(gdspath)
    cellname = cellname or temp_kcl.top_cell().name
    kcell = temp_kcl[cellname]
    c = kcell_to_component(kcell)
    if post_process:
        post_process(c)

    for shape in c.shapes(layer_pin).each(kf.kdb.Shapes.SPaths):
        path = shape.path
        p1, p2 = list(path.each_point())
        v = p2 - p1
        if v.x < 0:
            orientation = 2
        elif v.x > 0:
            orientation = 0
        elif v.y > 0:
            orientation = 1
        else:
            orientation = 3

        c.create_port(
            width=path.width,
            trans=kf.kdb.Trans(orientation, False, path.bbox().center().to_v()),
            layer=layer_port,
            port_type=port_type,
        )
    c.auto_rename_ports()
    c = kcell_to_component(c)
    c.function_name = cellname
    return c


if __name__ == "__main__":
    # from gdsfactory.write_cells import get_import_gds_script
    # script = get_import_gds_script(dirpath=PATH.gds, module="ubcpdk.components")
    # print(script)

    # gdsname = "ebeam_crossing4.gds"
    gdsname = "ebeam_y_1550.gds"
    c = gf.Component("my_component")
    wg1 = c << import_gds(gdsname)
    wg2 = c << import_gds(gdsname)
    wg2.dmove((100, 0))
    c.show()
