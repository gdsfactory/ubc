import gdsfactory as gf
from ubcpdk.import_gds import import_gds


@gf.cell
def gc_te1550() -> gf.Component:
    c = import_gds("ebeam_gc_te1550")
    c = c.rotate(angle=180)
    c.info.polarization = "te"
    c.info.wavelength = 1.550
    c.auto_rename_ports()
    return c


@gf.cell
def gc_te1550_broadband() -> gf.Component:
    c = import_gds("ebeam_gc_te1550_broadband")
    c = c.rotate(angle=180)
    c.info.polarization = "te"
    c.info.wavelength = 1.550
    c.auto_rename_ports()
    gf.port.auto_rename_ports(c)
    return c


@gf.cell
def gc_te1310() -> gf.Component:
    c = import_gds("ebeam_gc_te1310")
    c = c.rotate(angle=180)
    c.info.polarization = "te"
    c.info.wavelength = 1.310
    c.auto_rename_ports()
    return c


@gf.cell
def gc_tm1550() -> gf.Component:
    c = import_gds("ebeam_gc_tm1550")
    c = c.rotate(angle=180)
    c.info.polarization = "tm"
    c.info.wavelength = 1.550
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = gc_te1550()
    c.show()
