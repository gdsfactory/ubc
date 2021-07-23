from ubc.import_gds import import_gds
import pp
from pp.component import Component


@pp.cell_with_validator
def gc_te1550() -> Component:
    c = import_gds("ebeam_gc_te1550")
    c = pp.containers.rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1550
    pp.port.auto_rename_ports(c)
    return c


@pp.cell_with_validator
def gc_te1550_broadband() -> Component:
    c = import_gds("ebeam_gc_te1550_broadband")
    c = pp.containers.rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1550
    pp.port.auto_rename_ports(c)
    return c


@pp.cell_with_validator
def gc_te1310() -> Component:
    c = import_gds("ebeam_gc_te1310")
    c = pp.containers.rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1310
    pp.port.auto_rename_ports(c)
    return c


@pp.cell_with_validator
def gc_tm1550() -> Component:
    c = import_gds("ebeam_gc_tm1550")
    c = pp.containers.rotate(component=c, angle=180)
    c.polarization = "tm"
    c.wavelength = 1550
    pp.port.auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = gc_te1550()
    c.show()
