"""Tapers."""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from ubcpdk.tech import TECH


@gf.cell
def taper(
    length: float = 10.0,
    width1: float = TECH.width,
    width2: float | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        cross_section=cross_section,
        layer=None,
        port=None,
        with_two_ports=True,
        port_names=("o1", "o2"),
        port_types=("optical", "optical"),
        with_bbox=True,
    )


@gf.cell
def taper_metal(
    length: float = 10.0,
    width1: float = TECH.width_metal,
    width2: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> gf.Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        cross_section=cross_section,
        layer=None,
        port=None,
        with_two_ports=True,
        port_names=("e1", "e2"),
        port_types=("electrical", "electrical"),
        with_bbox=True,
    )
