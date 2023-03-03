"""Sample mask for the edx course Q1 2023."""

from typing import Optional, List

import ubcpdk
import ubcpdk.components as pdk
from ubcpdk.tech import LAYER
from ubcpdk.samples.write_mask import write_mask_gds_with_metadata

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring as _coupler_ring
from gdsfactory.components.via_stack import via_stack_heater_m3
from gdsfactory.components.straight import straight

from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Float2

via_stack_heater_m3_mini = gf.partial(via_stack_heater_m3, size=(4, 4))


size = (440, 470)
add_gc = ubcpdk.components.add_fiber_array

GC_PITCH = 127


@gf.cell
def ring_single_heater(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    coupler_ring: ComponentSpec = _coupler_ring,
    bend: ComponentSpec = bend_euler,
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    cross_section: CrossSectionSpec = "strip",
    via_stack: ComponentSpec = via_stack_heater_m3_mini,
    port_orientation: Optional[List[float]] = (180, 0),
    via_stack_offset: Float2 = (0, 0),
    **kwargs,
) -> gf.Component:
    """Override from gdsfactory to make ports face different directions.

    Returns a single ring with heater on top.

    ring coupler (cb: bottom) connects to two vertical straights (sl: left, sr: right),
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler function.
        bend: 90 degrees bend function.
        cross_section_waveguide_heater: for heater.
        cross_section: for regular waveguide.
        via_stack: for heater to routing metal.
        port_orientation: for electrical ports to promote from via_stack.
        via_stack_offset: x,y offset for via_stack.
        kwargs: cross_section settings.

    .. code::

          bl-st-br
          |      |
          sl     sr length_y
          |      |
         --==cb==-- gap

          length_x
    """
    gap = gf.snap.snap_to_grid(gap, nm=2)

    coupler_ring = gf.get_component(
        coupler_ring,
        bend=bend,
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        bend_cross_section=cross_section_waveguide_heater,
        **kwargs,
    )

    straight_side = straight(
        length=length_y,
        cross_section=cross_section_waveguide_heater,
        **kwargs,
    )
    straight_top = straight(
        length=length_x,
        cross_section=cross_section_waveguide_heater,
        **kwargs,
    )

    bend = gf.get_component(
        bend, radius=radius, cross_section=cross_section_waveguide_heater, **kwargs
    )

    c = gf.Component()
    cb = c << coupler_ring
    sl = c << straight_side
    sr = c << straight_side
    bl = c << bend
    br = c << bend
    st = c << straight_top

    sl.connect(port="o1", destination=cb.ports["o2"])
    bl.connect(port="o2", destination=sl.ports["o2"])

    st.connect(port="o2", destination=bl.ports["o1"])
    br.connect(port="o2", destination=st.ports["o1"])
    sr.connect(port="o1", destination=br.ports["o1"])
    sr.connect(port="o2", destination=cb.ports["o3"])

    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o1", port=cb.ports["o1"])

    via = gf.get_component(via_stack)
    c1 = c << via
    c2 = c << via
    c1.xmax = -length_x / 2 + cb.x - via_stack_offset[0]
    c2.xmin = +length_x / 2 + cb.x + via_stack_offset[0]
    c1.movey(via_stack_offset[1])
    c2.movey(via_stack_offset[1])
    c.add_ports(c1.get_ports_list(orientation=port_orientation[0]), prefix="e1")
    c.add_ports(c2.get_ports_list(orientation=port_orientation[1]), prefix="e2")
    c.auto_rename_ports()
    return c


@gf.cell
def rings_proximity(
    num_rings=5,
    sep_resonators=2,
    radius=10.0,
):
    """A sequence of multiple rings, with the first one having a heater."""
    c = gf.Component()
    gap = 0.2  # TODO: make variable
    width = 0.5  # TODO: make variable
    for index in range(num_rings):
        if index in [0, num_rings // 2]:
            ring = c << ring_single_heater(
                length_x=2, via_stack=pdk.via_stack_heater_mtop
            ).rotate(90).movex(-index * (sep_resonators + 2 * radius + 3 * width - gap))
            c.add_port(f"e1_{index}", port=ring.ports["e1"])
            c.add_port(f"e2_{index}", port=ring.ports["e2"])
        else:
            ring = c << gf.components.ring_single(length_x=2).rotate(90).movex(
                -index * (sep_resonators + 2 * radius + 3 * width - gap)
            )
        c.add_port(f"o1_{index}", port=ring.ports["o1"])
        c.add_port(f"o2_{index}", port=ring.ports["o2"])

    return c


@gf.cell
def disks_proximity(
    num_rings=5,
    sep_resonators=5,
    radius=10.0,
):
    c = gf.Component()
    gap = 0.2
    width = 0.5
    for index in range(num_rings):
        if index in [0, num_rings // 2]:
            disk = c << gf.components.disk_heater(
                wrap_angle_deg=10.0,
                radius=radius,
                port_orientation=270,
                via_stack=pdk.via_stack_heater_mtop,
                heater_layer=LAYER.M1_HEATER,
            ).rotate(90).movex(-index * (sep_resonators + 2 * radius + 2 * width + gap))
            c.add_port(f"e1_{index}", port=disk.ports["e2"])
            c.add_port(f"e2_{index}", port=disk.ports["e1"])
        else:
            disk = c << gf.components.disk(
                wrap_angle_deg=10.0,
                radius=radius,
            ).rotate(
                90
            ).movex(-index * (sep_resonators + 2 * radius + 2 * width + gap))
        c.add_port(f"o1_{index}", port=disk.ports["o1"])
        c.add_port(f"o2_{index}", port=disk.ports["o2"])
    return c


def bend_gc_array(
    gc_spec: ComponentSpec = pdk.gc_te1550(),
    bend_spec: ComponentSpec = gf.components.bend_euler(),
):
    """Two gc's with opposite bends.

    Not completely needed, was originally intended to make routing easier.
    """
    c = gf.Component()
    gc_top = c << gf.get_component(gc_spec).movey(GC_PITCH)
    bend_top = c << gf.get_component(bend_spec)
    bend_top.connect("o1", destination=gc_top.ports["o1"])

    gc_bot = c << gf.get_component(gc_spec)
    bend_bot = c << gf.get_component(bend_spec)
    bend_bot.connect("o2", destination=gc_bot.ports["o1"])

    c.add_port(name="o1", port=bend_top["o2"])
    c.add_port(name="o2", port=bend_bot["o1"])
    return c


def resonator_proximity_io(
    resonator_array: ComponentSpec = rings_proximity,
    num_resonators=9,
    sep_resonators=3,
    radius_resonators=10.0,
    grating_buffer=50.0,
    waveguide_buffer=2.5,
    gc_bus_buffer=10,
):
    """Resonator proximity experiment with fiber array.

    Arguments:
        resonator_array: component with resonator array (first one needs a heater)
        num_resonators, sep_resonators, radius_resonators: resonator_array arguments
        grating_buffer: distance between neighbouring grating couplers
        waveguide_buffer: distance between bus waveguides
        gc_bus_buffer: distance between the closest bus waveguide and grating coupler ports
    """
    c = gf.Component()
    resonators = c << resonator_array(
        num_rings=num_resonators,
        sep_resonators=sep_resonators,
        radius=radius_resonators,
    )
    resonators.movey(GC_PITCH / 2)
    for i in range(num_resonators + 1):
        gc_array = c << gf.get_component(bend_gc_array).movex(i * grating_buffer)
        gc_array.mirror()
        routes = []
        if i == 0:
            # Calibration, just add a waveguide
            routes.append(
                gf.routing.get_route(gc_array.ports["o1"], gc_array.ports["o2"])
            )
        else:
            # Route top ports to top GCs
            x0 = resonators.ports[f"o2_{i-1}"].x
            y0 = resonators.ports[f"o2_{i-1}"].y
            x2 = gc_array.ports["o1"].x
            y2 = gc_array.ports["o1"].y
            routes.append(
                gf.routing.get_route_from_waypoints(
                    [
                        (x0, y0),
                        (x0, y2 - gc_bus_buffer - waveguide_buffer * (i - 1)),
                        (x2, y2 - gc_bus_buffer - waveguide_buffer * (i - 1)),
                        (x2, y2),
                    ]
                )
            )
            # Route bottom ports to bottom GCs
            x0 = resonators.ports[f"o1_{i-1}"].x
            y0 = resonators.ports[f"o1_{i-1}"].y
            x2 = gc_array.ports["o2"].x
            y2 = gc_array.ports["o2"].y
            routes.append(
                gf.routing.get_route_from_waypoints(
                    [
                        (x0, y0),
                        (x0, y2 + gc_bus_buffer + waveguide_buffer * (i - 1)),
                        (x2, y2 + gc_bus_buffer + waveguide_buffer * (i - 1)),
                        (x2, y2),
                    ]
                )
            )
        for route in routes:
            c.add(route.references)

    c.add_port("e1", port=resonators.ports["e1"])
    c.add_port("e2", port=resonators.ports["e2"])

    return c


def needs_fixing():
    """Ring resonators with thermal cross-talk.

    Old cell; does not pass verification

    - needs labels.
    """
    c = gf.Component()
    rings = c << resonator_proximity_io(num_resonators=7)
    disks = c << resonator_proximity_io(
        resonator_array=disks_proximity, num_resonators=7
    ).movey(-GC_PITCH - 50)
    floorplan = c << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    floorplan.x = disks.xmin + size[0] / 2 - 25
    floorplan.y = disks.ymin + size[1] / 2 - 25

    # Add electrical pads
    pads = c << gf.get_component(
        gf.components.pad_array, columns=3, orientation=90
    ).rotate(90).movex(130).movey(-160)
    route = gf.routing.get_route_electrical(
        rings.ports["e1"], pads.ports["e13"], bend="wire_corner"
    )
    c.add(route.references)
    route = gf.routing.get_route_electrical(
        rings.ports["e2"], pads.ports["e12"], bend="wire_corner"
    )
    c.add(route.references)
    route = gf.routing.get_route_electrical(
        disks.ports["e1"], pads.ports["e12"], bend="wire_corner"
    )
    c.add(route.references)
    route = gf.routing.get_route_electrical(
        disks.ports["e2"], pads.ports["e11"], bend="wire_corner"
    )
    c.add(route.references)

    return write_mask_gds_with_metadata(c)


def test_mask1():
    """Ring resonators with thermal cross-talk.

    Old cell; does not pass verification
    """
    rings_active = [pdk.ring_single_heater(length_x=4)]
    rings_passive = [pdk.ring_single(length_x=4)]

    rings_passive = [gf.functions.rotate180(ring) for ring in rings_passive]
    rings_active = [gf.functions.rotate180(ring) for ring in rings_active]

    rings_active_gc = [pdk.add_fiber_array_pads_rf(ring) for ring in rings_active]
    rings_passive_gc = [pdk.add_fiber_array(ring) for ring in rings_passive]
    rings_gc = rings_passive_gc + rings_active_gc

    m = gf.Component()
    spacing = 1
    g = m << gf.grid(
        rings_gc,
        shape=(1, len(rings_gc)),
        spacing=(spacing, spacing),
        add_ports_prefix=False,
        add_ports_suffix=True,
    )
    g.xmin = 1
    g.ymin = 1

    m.add_ports(g.ports)
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    m.name = "EBeam_JoaquinMatres_Simon_0"
    return write_mask_gds_with_metadata(m)


def crosstalk_experiment_parametrized_mask(
    name="EBeam_JoaquinMatres_Simon_1",
    num_gcs: int = 10,
    num_gc_per_pitch: int = 5,
    sep_resonators: float = 15.0,
    ring_y_offset: float = 0.0,
    resonator_func: ComponentSpec = rings_proximity,
    fill_layers=None,
    fill_margin=2,
    fill_size=(0.5, 0.5),
    padding=20,
):
    """Ring resonators with thermal cross-talk.

    name: for labels
    num_gcs: number of grating couplers (should be <10)
    num_gc_per_pitch: number of grating couplers within a GC pitch (5 is optimal)
    sep_resonators: distance between the resonators
    ring_y_offset: manual offset for the resonator positions to make the routes DRC clean
    resonator_func: rings_proximity or disks_proximity
    fill_layers: layers to add as unity dennity fill around the rings
    fill_margin: keepout between the fill_layers and the same design layers
    fill_size: tiling size
    padding: how much to extend the fill beyond the ring component
    """
    m = gf.Component()

    # GC array
    spacing = GC_PITCH / num_gc_per_pitch - (
        pdk.gc_te1550().ymax - pdk.gc_te1550().ymin
    )
    g = m << gf.grid(
        [pdk.gc_te1550()] * num_gcs,
        shape=(num_gcs, 1),
        spacing=(spacing, spacing),
        add_ports_prefix=False,
        add_ports_suffix=True,
        rotation=180,
    )
    g.xmin = 25
    g.ymin = 150

    # Pads
    pad_spacing = 125 - (pdk.pad().ymax - pdk.pad().ymin)
    pads = m << gf.grid(
        [pdk.pad] * 4,
        shape=(4, 1),
        spacing=(pad_spacing, pad_spacing),
        add_ports_prefix=False,
        add_ports_suffix=True,
    )
    pads.xmin = 360
    pads.ymin = 10

    # Rings
    rings_component = (
        resonator_func(num_rings=num_gcs // 2, sep_resonators=sep_resonators)
        .rotate(90)
        .movex(g.xmin + 225)
        .movey((pads.ymin + pads.ymax) / 2 + ring_y_offset)
    )
    if fill_layers:
        for layer in fill_layers:
            m << gf.fill_rectangle(
                rings_component,
                fill_size=fill_size,
                fill_layers=[layer],
                margin=fill_margin,
                fill_densities=[1.0],
                avoid_layers=[layer],
            )
    rings = m << rings_component

    # Left optical connections
    right_ports = [rings.ports[f"o2_{i}"] for i in range(num_gc_per_pitch)]
    left_ports = [g.ports[f"o1_{i}_0"] for i in range(num_gc_per_pitch)]
    routes = gf.routing.get_bundle(right_ports, left_ports)
    for route in routes:
        m.add(route.references)

    # GC loopbacks for easier routing
    extended_gc_ports = []
    for i in range(num_gc_per_pitch, num_gcs - 1):
        bend = m << gf.get_component(gf.components.bend_euler180)
        bend.connect("o2", destination=g.ports[f"o1_{i}_0"])
        escape = (
            m
            << gf.get_component(
                gf.components.bezier,
                control_points=[(0.0, 0.0), (15.0, 0.0), (15.0, 7.5), (30.0, 7.5)],
            ).mirror()
        )
        escape.connect("o1", destination=bend.ports["o1"])
        straight = m << gf.get_component(gf.components.straight, length=35 - 4 * i)
        straight.connect("o1", destination=escape.ports["o2"])
        bend = m << gf.get_component(gf.components.bend_euler)
        bend.connect("o1", destination=straight.ports["o2"])
        extended_gc_ports.append(bend.ports["o2"])
    bend = m << gf.get_component(gf.components.bend_euler)
    bend.connect("o2", destination=g.ports[f"o1_{num_gcs-1}_0"])
    extended_gc_ports.append(bend.ports["o1"])

    # Right optical connections
    right_ports = [rings.ports[f"o1_{i}"] for i in range(num_gc_per_pitch)]
    left_ports = extended_gc_ports
    for i, (port1, port2) in enumerate(zip(right_ports, left_ports)):
        x0 = port1.x
        y0 = port1.y
        x2 = port2.x
        y2 = port2.y
        dx = 50 + (len(right_ports) - i) * 5
        y1 = rings.ymin - 50 - (len(right_ports) - i) * 5
        route = gf.routing.get_route_from_waypoints(
            [(x0, y0), (x0 + dx, y0), (x0 + dx, y1), (x2, y1), (x2, y2)]
        )
        m.add(route.references)

    # Electrical connections
    for ring_index, pad_index in zip([0, num_gcs // 4], [0, 3]):
        ring_port = rings.ports[f"e2_{ring_index}"]
        pad_port = pads.ports[f"e1_{pad_index}_0"]
        x0 = ring_port.x
        y0 = ring_port.y
        x2 = pad_port.x
        y2 = pad_port.y
        dx = -50
        route = gf.routing.get_route_from_waypoints(
            [(x0, y0), (x0 + dx, y0), (x0 + dx, y2), (x2, y2)],
            cross_section="metal_routing",
            bend=gf.components.wire_corner,
        )
        m.add(route.references)
    for ring_index, pad_index in zip([0, num_gcs // 4], [1, 2]):
        ring_port = rings.ports[f"e1_{ring_index}"]
        pad_port = pads.ports[f"e1_{pad_index}_0"]
        x0 = ring_port.x
        y0 = ring_port.y
        x2 = pad_port.x
        y2 = pad_port.y
        dx = 50
        route = gf.routing.get_route_from_waypoints(
            [(x0, y0), (x0 + dx, y0), (x0 + dx, y2), (x2, y2)],
            cross_section="metal_routing",
            bend=gf.components.wire_corner,
        )
        m.add(route.references)

    # Add test labels
    # For every experiment, label the input GC (bottom one)
    for i in range(num_gc_per_pitch, num_gcs):
        unique_name = f"opt_in_TE_1550_device_{name}_{i}"
        # Place label at GC port
        label = gf.component_layout.Label(
            text=unique_name,
            origin=g.ports[f"o1_{i}_0"].center,
            anchor="o",
            magnification=1.0,
            rotation=0.0,
            layer=LAYER.LABEL[0],
            texttype=LAYER.LABEL[1],
            x_reflection=False,
        )
        m.add(label)
        # Place label at electrical ports
        for index, padname in enumerate(["G1", "S1", "G2", "S2"][::-1]):
            label = gf.component_layout.Label(
                text=f"elec_{unique_name}_{padname}",
                origin=(pads.xmin + 75 / 2, pads.ymin + (125) * index + 75 / 2),
                anchor="o",
                magnification=1.0,
                rotation=0.0,
                layer=LAYER.LABEL[0],
                texttype=LAYER.LABEL[1],
                x_reflection=False,
            )
            m.add(label)

    m.add_ports(g.ports)
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    m.name = name
    return m


def test_mask3():
    """Rings with thermal crosstalk, close rings"""
    m = crosstalk_experiment_parametrized_mask(
        name="EBeam_JoaquinMatres_Simon_1",
        sep_resonators=5.0,
        ring_y_offset=20.0,
        resonator_func=rings_proximity,
    )
    return write_mask_gds_with_metadata(m)


def test_mask4():
    """Rings with thermal crosstalk, far rings"""
    m = crosstalk_experiment_parametrized_mask(
        name="EBeam_JoaquinMatres_Simon_2",
        sep_resonators=20.0,
        ring_y_offset=40.0,
        resonator_func=rings_proximity,
    )
    return write_mask_gds_with_metadata(m)


def test_mask5():
    """Rings with thermal crosstalk, metal fill"""
    m = crosstalk_experiment_parametrized_mask(
        name="EBeam_JoaquinMatres_Simon_3",
        sep_resonators=20.0,
        ring_y_offset=40.0,
        resonator_func=rings_proximity,
        fill_layers=[LAYER.M1_HEATER],
    )
    return write_mask_gds_with_metadata(m)


def test_mask6():
    """Rings with thermal crosstalk, silicon fill"""
    m = crosstalk_experiment_parametrized_mask(
        name="EBeam_JoaquinMatres_Simon_4",
        sep_resonators=20.0,
        ring_y_offset=40.0,
        resonator_func=rings_proximity,
        fill_layers=[LAYER.WG, LAYER.M1_HEATER],
        fill_margin=5,
        fill_size=(0.5, 0.5),
    )
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    m, _ = test_mask1()
    # m, _ = test_mask3()
    # m, _ = test_mask4()
    # m, _ = test_mask5()
    m.show()
