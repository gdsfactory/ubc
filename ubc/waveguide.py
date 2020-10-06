import pp
from pp.component import Component
from ubc.add_pins import add_pins
from ubc.layers import LAYER


@pp.autoname
def waveguide(length: float = 10, width: float = 0.5) -> Component:
    """ straight waveguide """
    c = pp.c.waveguide(
        length=length,
        width=width,
        layer=LAYER.WG,
        layers_cladding=[LAYER.DEVREC],
        cladding_offset=1,
    )
    labels = [
        "Lumerical_INTERCONNECT_library=Design kits/EBeam",
        "Lumerical_INTERCONNECT_component=ebeam_wg_integral_1550",
        f"Spice_param:wg_width={width:.3f}u wg_length={length:.3f}u",
    ]

    for i, text in enumerate(labels):
        c.add(pp.c.label(text=text, position=(length / 2, i * 0.1), layer=LAYER.DEVREC))
    add_pins(c)
    return c


if __name__ == "__main__":
    c = waveguide()
    pp.show(c)
