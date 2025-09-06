"""Evanescent Couplers."""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from ubcpdk.tech import TECH


@gf.cell
def coupler(length: float = 14.5, gap: float = TECH.gap) -> gf.Component:
    """Returns Symmetric coupler.

    Args:
        length: of coupling region in um.
        gap: of coupling region in um.
    """
    return gf.c.coupler(
        length=length,
        gap=gap,
        dy=4.0,
        dx=10.0,
        cross_section="strip",
        allow_min_radius_violation=False,
    )


@gf.cell
def coupler_ring(
    length_x: float = 4,
    gap: float = TECH.gap,
    radius: float = TECH.radius,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: str = "strip",
    cross_section_bend: CrossSectionSpec | None = None,
    length_extension: float = 10,
) -> gf.Component:
    """Returns Coupler for ring.

    Args:
        length_x: length of the parallel coupled straight waveguides.
        gap: gap between for coupler.
        radius: for the bend and coupler.
        bend: 90 degrees bend spec.
        straight: straight spec.
        cross_section: cross_section spec.
        cross_section_bend: cross_section for the bend. Defaults to cross_section.
        length_extension: length extension for the coupler.
    """
    return gf.c.coupler_ring(
        length_x=length_x,
        gap=gap,
        radius=radius,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        cross_section_bend=cross_section_bend,
        length_extension=length_extension,
    )
