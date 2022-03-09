import gdsfactory as gf

from ubcpdk.tech import LAYER
from gdsfactory.add_pins import add_pins_bbox_siepic


@gf.cell
def straight(
    length: float = 10.0,
    width: float = 0.5,
    layer: gf.types.Layer = LAYER.WG,
    with_pins: bool = True,
    **kwargs,
) -> gf.Component:
    """Straight waveguide.

    Args:
        length:
        width:
        layer:
        with_pins:
    """
    c = gf.Component()

    s = gf.components.straight(length=length, width=width, layer=layer, **kwargs)
    ref = c << s
    c.add_ports(ref.ports)
    c.copy_child_info(s)

    if with_pins:
        labels = [
            "Lumerical_INTERCONNECT_library=Design kits/EBeam",
            "Lumerical_INTERCONNECT_component=ebeam_wg_integral_1550",
            f"Spice_param:wg_width={width:.3f}u wg_length={length:.3f}u",
        ]

        for i, text in enumerate(labels):
            c.add_label(text=text, position=(length / 2, i * 0.1), layer=LAYER.DEVREC)
        add_pins_bbox_siepic(c)
    return c


if __name__ == "__main__":
    c = gf.Component()
    # s1 = c.add_ref(straight())

    s1 = c << straight()
    s2 = c << straight()
    s2.connect("o2", s1.ports["o1"])
    c.show(show_ports=False)
