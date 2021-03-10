import pp
from pp.component import Component
from pp.tech import Tech
from pp.types import Layer
from ubc.add_pins import add_pins
from ubc.layers import LAYER
from ubc.tech import TECH_SILICON_C


@pp.cell
def waveguide(
    length: float = 10.0,
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    tech: Tech = TECH_SILICON_C,
    **kwargs,
) -> Component:
    """Straight waveguide."""
    c = pp.c.waveguide(length=length, width=width, layer=layer, tech=tech, **kwargs)
    labels = [
        "Lumerical_INTERCONNECT_library=Design kits/EBeam",
        "Lumerical_INTERCONNECT_component=ebeam_wg_integral_1550",
        f"Spice_param:wg_width={width:.3f}u wg_length={length:.3f}u",
    ]

    for i, text in enumerate(labels):
        c.add(pp.c.label(text=text, position=(length / 2, i * 0.1), layer=LAYER.DEVREC))
    c = add_pins(c)
    return c


if __name__ == "__main__":
    c = waveguide()
    c.show()
