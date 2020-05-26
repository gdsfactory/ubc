import pp
from ubc.layers import LAYER


@pp.autoname
def waveguide(length=10, width=0.5):
    c = pp.c.waveguide(
        length=length,
        width=width,
        layer=LAYER.WG,
        layers_cladding=[LAYER.DEVREC],
        with_pins=True,
    )
    labels = [
        f"Lumerical_INTERCONNECT_library=Design kits/EBeam",
        f"Lumerical_INTERCONNECT_component=ebeam_wg_integral_1550",
        f"Spice_param:wg_width={width:.3f}u wg_length={length:.3f}u",
    ]

    for i, text in enumerate(labels):
        c.add(pp.c.label(text=text, position=(length / 2, i * 0.1), layer=LAYER.DEVREC))
    return c


if __name__ == "__main__":
    c = waveguide()
    # pp.show(c)
