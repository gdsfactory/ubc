import gdsfactory as gf
from ubc.import_gds import import_gds


@gf.cell
def gc_te1550() -> gf.Component:
    c = import_gds("ebeam_gc_te1550")
    c = gf.containers.rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1550
    gf.port.auto_rename_ports(c)
    return c


@gf.cell
def gc_te1550_broadband() -> gf.Component:
    c = import_gds("ebeam_gc_te1550_broadband")
    c = gf.containers.rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1550
    gf.port.auto_rename_ports(c)
    return c


@gf.cell
def gc_te1310() -> gf.Component:
    c = import_gds("ebeam_gc_te1310")
    c = gf.containers.rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1310
    gf.port.auto_rename_ports(c)
    return c


@gf.cell
def gc_tm1550() -> gf.Component:
    c = import_gds("ebeam_gc_tm1550")
    c = gf.containers.rotate(component=c, angle=180)
    c.polarization = "tm"
    c.wavelength = 1550
    gf.port.auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = gc_te1550()
    c.show()
