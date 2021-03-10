import pp
from pp.component import Component
from pp.tech import Tech
from pp.types import Layer
from ubc.add_pins import add_pins
from ubc.layers import LAYER
from ubc.tech import TECH_SILICON_C


@pp.cell
def bend90(
    radius: int = 10,
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    tech: Tech = TECH_SILICON_C,
    **kwargs,
) -> Component:
    c = pp.c.bend_circular(radius=radius, width=width, layer=layer, tech=tech)
    labels = [
        "Lumerical_INTERCONNECT_library=Design kits/EBeam",
        "Lumerical_INTERCONNECT_component=ebeam_bend_1550",
        f"Spice_param:radius={radius:.3f}u wg_width={width:.3f}u",
    ]

    for i, text in enumerate(labels):
        c.add(pp.c.label(text=text, position=(c.x, c.y + i * 0.1), layer=LAYER.DEVREC))
    c = add_pins(c)
    return c


if __name__ == "__main__":
    c = bend90()
    c.show()
