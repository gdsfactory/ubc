"""Ring Resonators."""

import gdsfactory as gf
from gdsfactory.typings import AngleInDegrees, ComponentSpec, CrossSectionSpec, Float2

from ubcpdk.tech import TECH


@gf.cell
def ring_single(
    gap: float = TECH.gap_strip,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    cross_section: CrossSectionSpec = "strip",
    length_extension: float = 10.0,
) -> gf.Component:
    """Returns a single ring.

    ring coupler (cb: bottom) connects to two vertical straights (sl: left, sr: right),
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        bend: 90 degrees bend spec.
        straight: straight spec.
        coupler_ring: ring coupler spec.
        cross_section: cross_section spec.
        length_extension: straight length extension at the end of the coupler bottom ports.


    .. code::

                    xxxxxxxxxxxxx
                xxxxx           xxxx
              xxx                   xxx
            xxx                       xxx
           xx                           xxx
           x                             xxx
          xx                              xx▲
          xx                              xx│length_y
          xx                              xx▼
          xx                             xx
           xx          length_x          x
            xx     ◄───────────────►    x
             xx                       xxx
               xx                   xxx
                xxx──────▲─────────xxx
                         │gap
                 o1──────▼─────────o2
    """
    return gf.c.ring_single(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend="bend_euler",
        straight="straight",
        coupler_ring="coupler_ring",
        cross_section=cross_section,
        length_extension=length_extension,
    )


@gf.cell
def ring_double(
    gap: float = TECH.gap_strip,
    gap_top: float | None = None,
    gap_bot: float | None = None,
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
    cross_section: CrossSectionSpec = "strip",
    length_extension: float = 10.0,
) -> gf.Component:
    """Returns a double bus ring.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        gap_top: gap for the top coupler. Defaults to gap.
        gap_bot: gap for the bottom coupler. Defaults to gap.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        bend: 90 degrees bend spec.
        straight: straight spec.
        coupler_ring: ring coupler spec.
        coupler_ring_top: top ring coupler spec. Defaults to coupler_ring.
        cross_section: cross_section spec.
        length_extension: straight length extension at the end of the coupler bottom ports.
    """
    return gf.c.ring_double(
        gap=gap,
        gap_top=gap_top,
        gap_bot=gap_bot,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend="bend_euler",
        straight="straight",
        coupler_ring="coupler_ring",
        cross_section=cross_section,
        length_extension=length_extension,
    )


@gf.cell
def ring_double_heater(
    gap: float = 0.2,
    gap_top: float | None = None,
    gap_bot: float | None = None,
    radius: float = 10.0,
    length_x: float = 2.0,
    length_y: float = 0.01,
    coupler_ring: ComponentSpec = "coupler_ring",
    coupler_ring_top: ComponentSpec | None = None,
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    cross_section: CrossSectionSpec = "strip",
    via_stack: ComponentSpec = "via_stack_heater_mtop",
    port_orientation: AngleInDegrees | None = None,
    via_stack_offset: Float2 = (1, 0),
    via_stack_size: Float2 = (4, 4),
    length_extension: float | None = None,
) -> gf.Component:
    """Returns a double bus ring with heater on top.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        gap_top: gap for the top coupler. Defaults to gap.
        gap_bot: gap for the bottom coupler. Defaults to gap.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        coupler_ring_top: ring coupler spec for coupler away from vias (defaults to coupler_ring)
        straight: straight spec.
        bend: bend spec.
        cross_section_heater: for heater.
        cross_section_waveguide_heater: for waveguide with heater.
        cross_section: for regular waveguide.
        via_stack: for heater to routing metal.
        port_orientation: for electrical ports to promote from via_stack.
        via_stack_size: size of via_stack.
        via_stack_offset: x,y offset for via_stack.
        with_drop: adds drop ports.
        length_extension: straight length extension at the end of the coupler bottom ports. None defaults to 3.0 + radius.

    .. code::

           o2──────▲─────────o3
                   │gap_top
           xx──────▼─────────xxx
          xxx                   xxx
        xxx                       xxx
       xx                           xxx
       x                             xxx
      xx                              xx▲
      xx                              xx│length_y
      xx                              xx▼
      xx                             xx
       xx          length_x          x
        xx     ◄───────────────►    x
         xx                       xxx
           xx                   xxx
            xxx──────▲─────────xxx
                     │gap
             o1──────▼─────────o4
    """
    return gf.c.ring_single_heater(
        gap=gap,
        gap_top=gap_top,
        gap_bot=gap_bot,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        coupler_ring=coupler_ring,
        coupler_ring_top=coupler_ring_top,
        straight=straight,
        bend=bend,
        cross_section_heater=cross_section_heater,
        cross_section_waveguide_heater=cross_section_waveguide_heater,
        cross_section=cross_section,
        via_stack=via_stack,
        port_orientation=port_orientation,
        via_stack_offset=via_stack_offset,
        with_drop=True,
        length_extension=length_extension,
        via_stack_size=via_stack_size,
    )


@gf.cell
def ring_single_heater(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 2.0,
    length_y: float = 0.01,
    coupler_ring: ComponentSpec = "coupler_ring",
    coupler_ring_top: ComponentSpec | None = None,
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    cross_section: CrossSectionSpec = "strip",
    via_stack: ComponentSpec = "via_stack_heater_mtop",
    port_orientation: AngleInDegrees | None = None,
    via_stack_offset: Float2 = (1, 0),
    via_stack_size: Float2 = (4, 4),
    length_extension: float | None = None,
) -> gf.Component:
    """Returns a double bus ring with heater on top.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        gap_top: gap for the top coupler. Defaults to gap.
        gap_bot: gap for the bottom coupler. Defaults to gap.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        coupler_ring_top: ring coupler spec for coupler away from vias (defaults to coupler_ring)
        straight: straight spec.
        bend: bend spec.
        cross_section_heater: for heater.
        cross_section_waveguide_heater: for waveguide with heater.
        cross_section: for regular waveguide.
        via_stack: for heater to routing metal.
        port_orientation: for electrical ports to promote from via_stack.
        via_stack_size: size of via_stack.
        via_stack_offset: x,y offset for via_stack.
        with_drop: adds drop ports.
        length_extension: straight length extension at the end of the coupler bottom ports. None defaults to 3.0 + radius.

    .. code::

               xxx          xx
            xxx               xxx
          xxx                   xxx
        xxx                       xxx
       xx                           xxx
       x                             xxx
      xx                              xx▲
      xx                              xx│length_y
      xx                              xx▼
      xx                             xx
       xx          length_x          x
        xx     ◄───────────────►    x
         xx                       xxx
           xx                   xxx
            xxx──────▲─────────xxx
                     │gap
             o1──────▼─────────o4
    """
    return gf.c.ring_single_heater(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        coupler_ring=coupler_ring,
        coupler_ring_top=coupler_ring_top,
        straight=straight,
        bend=bend,
        cross_section_heater=cross_section_heater,
        cross_section_waveguide_heater=cross_section_waveguide_heater,
        cross_section=cross_section,
        via_stack=via_stack,
        port_orientation=port_orientation,
        via_stack_offset=via_stack_offset,
        with_drop=False,
        length_extension=length_extension,
        via_stack_size=via_stack_size,
    )


if __name__ == "__main__":
    from ubcpdk import PDK

    PDK.activate()

    # c = add_fiber_array("ring_double")
    # c =gf.get_component(gc)
    # c = pack_doe()
    c = ring_single_heater()
    c.pprint_ports()
    c.show()
