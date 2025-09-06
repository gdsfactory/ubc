from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import gdsfactory as gf
import kfactory as kf
from gdsfactory.component import Component
from kfactory import KCLayout

from ubcpdk.tech import LAYER


def kcell_to_component(
    kcell: kf.kcell.ProtoTKCell[Any], load_meta: bool = False
) -> Component:
    kcell.set_meta_data()

    if load_meta:
        for ci in kcell.called_cells():
            kcell.kcl[ci].set_meta_data()

    c = Component()
    c.name = kcell.name
    c.kdb_cell.copy_tree(kcell.kdb_cell)
    if load_meta:
        c.copy_meta_info(kcell.kdb_cell)
        c.get_meta_data()

    for ci in c.called_cells():
        c.kcl[ci].get_meta_data()

    return c


@cache
def import_gds(
    gdspath: str | Path,
    cellname: str | None = None,
    port_type="optical",
    layer_pin=LAYER.PORT,
    layer_port=LAYER.WG,
    post_process: Callable[[Component], Component] | None = None,
    convert_paths_to_polygons: bool = True,
    auto_rename_ports: bool = True,
) -> Component:
    """Returns klayout cell from GDS.

    Args:
        gdspath: path to gds.
        cellname: cell name. If None uses top cell.
        port_type: port type.
        layer_pin: layer where pins are drawn.
        layer_port: layer where ports are drawn.
        post_process: function to apply to component after import.
        convert_paths_to_polygons: convert paths to polygons.
        auto_rename_ports: rename ports.
    """
    temp_kcl = KCLayout(name=str(gdspath))
    temp_kcl.read(gdspath)
    cellname = cellname or temp_kcl.top_cell().name
    kcell = temp_kcl[cellname]

    c = kcell_to_component(kcell)
    if post_process:
        post_process(c)

    layer_pin = gf.get_layer_info(layer_pin)

    for shape in c.shapes(layer_pin).each(kf.kdb.Shapes.SPaths):
        path = shape.path
        assert isinstance(path, kf.kdb.Path)
        dpath = path.to_dtype(gf.kcl.dbu)
        p1, p2 = list(dpath.each_point())
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
            width=gf.snap.snap_to_grid(path.width / 1e3, nm=2),
            trans=kf.kdb.Trans(orientation, False, path.bbox().center().to_v()),
            layer=layer_port,
            port_type=port_type,
        )

    if convert_paths_to_polygons:
        for layer in c.kcl.layer_indexes():
            paths = list(c.shapes(layer).each(kf.kdb.Shapes.SPaths))

            for shape in paths:
                poly = shape.path.polygon()
                c.shapes(layer).erase(shape)
                c.shapes(layer).insert(poly)

    if auto_rename_ports:
        c.auto_rename_ports()
    c.function_name = cellname
    return c


if __name__ == "__main__":
    from ubcpdk.config import PATH

    c = import_gds(PATH.gds / "ebeam_gc_te1310_8deg.gds")
    c.pprint_ports()
    c.show()
