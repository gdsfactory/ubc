"""Grating couplers."""

import gdsfactory as gf


@gf.cell
def grating_coupler_elliptical(
    wavelength: float = 1.55,
    grating_line_width=0.315,
    cross_section="strip",
) -> gf.Component:
    """A grating coupler with curved but parallel teeth.

    Args:
        wavelength: the center wavelength for which the grating is designed
        grating_line_width: the line width of the grating
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_elliptical(
        polarization="te",
        wavelength=wavelength,
        grating_line_width=grating_line_width,
        taper_length=16.0,
        taper_angle=30.0,
        fiber_angle=15.0,
        neff=2.638,
        layer_slab=None,
        n_periods=30,
        cross_section=cross_section,
    )


if __name__ == "__main__":
    from ubcpdk import PDK

    PDK.activate()
    c = grating_coupler_elliptical()
    c.show()
