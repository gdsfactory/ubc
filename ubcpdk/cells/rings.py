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
    if length_y == 0 or length_x == 0:
        # Handle zero-length straights specially to avoid port overlap issues
        c = gf.Component()

        settings = {
            "gap": gap,
            "radius": radius,
            "length_x": length_x,
            "cross_section": cross_section,
            "bend": "bend_euler",
            "straight": "straight",
            "length_extension": length_extension,
        }

        coupler = gf.get_component("coupler_ring", settings=settings)
        cb = c << coupler

        b = gf.get_component("bend_euler", cross_section=cross_section, radius=radius)
        bl = c << b  # Left bend
        br = c << b  # Right bend

        if length_y == 0 and length_x == 0:
            # Both zero: connect bends directly to coupler and to each other
            bl.connect(port="o2", other=cb.ports["o2"])
            br.connect(port="o2", other=bl.ports["o1"])
            br.connect(port="o1", other=cb.ports["o3"])
        elif length_y == 0:
            # Only length_y=0: skip vertical straights, keep horizontal
            sx = gf.get_component(
                "straight", length=length_x, cross_section=cross_section
            )
            st = c << sx
            bl.connect(port="o2", other=cb.ports["o2"])
            st.connect(port="o2", other=bl.ports["o1"])
            br.connect(port="o2", other=st.ports["o1"])
            br.connect(port="o1", other=cb.ports["o3"])
        else:
            # Only length_x=0: skip horizontal straight, keep vertical
            sy = gf.get_component(
                "straight", length=length_y, cross_section=cross_section
            )
            sl = c << sy
            sr = c << sy
            sl.connect(port="o1", other=cb.ports["o2"])
            bl.connect(port="o2", other=sl.ports["o2"])
            br.connect(port="o2", other=bl.ports["o1"])
            sr.connect(port="o1", other=br.ports["o1"])
            sr.connect(port="o2", other=cb.ports["o3"])

        c.add_port("o2", port=cb.ports["o4"])
        c.add_port("o1", port=cb.ports["o1"])
        c.info["radius"] = coupler.info["radius"]
        return c

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
