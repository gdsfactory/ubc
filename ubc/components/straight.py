import pp

from ubc.tech import LAYER
from ubc.add_pins import add_pins


@pp.cell_with_validator
def straight(
    length: float = 10.0,
    width: float = 0.5,
    layer: pp.types.Layer = LAYER.WG,
    with_pins: bool = True,
    **kwargs,
) -> pp.Component:
    """Straight waveguide."""
    c = pp.components.straight(length=length, width=width, layer=layer, **kwargs)

    if with_pins:
        labels = [
            "Lumerical_INTERCONNECT_library=Design kits/EBeam",
            "Lumerical_INTERCONNECT_component=ebeam_wg_integral_1550",
            f"Spice_param:wg_width={width:.3f}u wg_length={length:.3f}u",
        ]

        for i, text in enumerate(labels):
            c.add_label(text=text, position=(length / 2, i * 0.1), layer=LAYER.DEVREC)
        add_pins(c)
    return c


if __name__ == "__main__":
    c = straight()
    c.show()
