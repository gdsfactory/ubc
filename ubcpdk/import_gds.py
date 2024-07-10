from functools import cache

import gdsfactory as gf
import kfactory as kf
from gdsfactory.component import Component
from gdsfactory.read.import_gds import kcell_to_component

from ubcpdk.config import PATH
from ubcpdk.tech import LAYER


@cache
def import_gds(
    name, port_type="optical", layer_pin=LAYER.PORT, layer_port=LAYER.WG
) -> Component:
    """Returns klayout cell from GDS."""
    kcl = kf.KCLayout(name=name)
    kcl.read(PATH.gds / name)
    top_cell = kcl.top_cell()
    c = kf.KCell(top_cell.name)
    c.copy_tree(top_cell)
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
    return kcell_to_component(c)


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
