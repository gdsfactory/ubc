"""Sample mask for the edx course Q1 2023."""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec

import ubcpdk
import ubcpdk.components as pdk
from ubcpdk.tech import LAYER
from ubcpdk.samples.write_mask import write_mask_gds_with_metadata


size = (440, 470)
add_gc = ubcpdk.components.add_fiber_array

GC_PITCH = 127


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
        if index == 0:
            ring = c << pdk.ring_single_heater(
                length_x=2, port_orientation=270, via_stack=pdk.via_stack_heater_mtop
            ).rotate(90).movex(-index * (sep_resonators + 2 * radius + 3 * width - gap))
            c.add_port("e1", port=ring.ports["e1"])
            c.add_port("e2", port=ring.ports["e2"])
        else:
            ring = c << gf.components.ring_single(length_x=2).rotate(90).movex(
                -index * (sep_resonators + 2 * radius + 3 * width - gap)
            )
        c.add_port(f"o1_{index}", port=ring.ports["o1"])
        c.add_port(f"o2_{index}", port=ring.ports["o2"])
    return c


def disks_proximity(
    num_rings=5,
    sep_resonators=5,
    radius=10.0,
):
    c = gf.Component()
    gap = 0.2
    width = 0.5
    for index in range(num_rings):
        if index == 0:
            disk = c << gf.components.disk_heater(
                wrap_angle_deg=10.0,
                radius=radius,
                port_orientation=270,
                via_stack=pdk.via_stack_heater_mtop,
                heater_layer=LAYER.M1_HEATER,
            ).rotate(90).movex(-index * (sep_resonators + 2 * radius + 2 * width + gap))
            c.add_port("e1", port=disk.ports["e1"])
            c.add_port("e2", port=disk.ports["e2"])
        else:
            disk = c << gf.components.disk(wrap_angle_deg=10.0, radius=radius,).rotate(
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


def test_mask0():
    """Ring resonators with thermal cross-talk.

    TODO: does not pass verification.

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
    """Ring resonators with thermal cross-talk."""
    rings_active = [pdk.ring_single_heater(length_x=4)]
    rings_passive = [pdk.ring_single(length_x=4)] * 2
    rings_active_gc = [pdk.add_fiber_array_pads_rf(ring) for ring in rings_active]
    rings_passive_gc = [pdk.add_fiber_array(ring) for ring in rings_passive]
    rings_gc = rings_passive_gc + rings_active_gc

    m = gf.Component()
    spacing = -20
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
    m.name = "EBeam_JoaquinMatres_Simon_1"
    return write_mask_gds_with_metadata(m)


def test_mask2():
    """Ring resonators with thermal cross-talk."""
    m = gf.Component()

    # GC array
    num_gcs = 10
    num_gc_per_pitch = 5
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
    g.xmin = 30
    g.ymin = 150

    # Rings10
    rings = m << rings_proximity(num_rings=num_gcs // 2, sep_resonators=15).rotate(
        90
    ).movex(g.xmin + 175).movey(300)

    # Pads
    pad_spacing = 125 - (pdk.pad().ymax - pdk.pad().ymin)
    pads = m << gf.grid(
        [pdk.pad] * 4,
        shape=(4, 1),
        spacing=(pad_spacing, pad_spacing),
        add_ports_prefix=False,
        add_ports_suffix=True,
    )
    pads.xmin = 350
    pads.ymin = 10

    # Optical connections
    right_ports = [rings.ports[f"o2_{i}"] for i in range(num_gc_per_pitch)]
    left_ports = [g.ports[f"o1_{i}_0"] for i in range(num_gc_per_pitch)]
    routes = gf.routing.get_bundle(right_ports, left_ports)
    for route in routes:
        m.add(route.references)

    # GC loopbacks
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

    right_ports = [rings.ports[f"o1_{i}"] for i in range(num_gc_per_pitch)]
    left_ports = extended_gc_ports
    for i, (port1, port2) in enumerate(zip(right_ports, left_ports)):
        print(i, port1, port2)
        x0 = port1.x
        y0 = port1.y
        x2 = port2.x
        y2 = port2.y
        dx = 50 + (len(right_ports) - i) * 5
        y1 = 100 - (len(right_ports) - i) * 5
        route = gf.routing.get_route_from_waypoints(
            [(x0, y0), (x0 + dx, y0), (x0 + dx, y1), (x2, y1), (x2, y2)]
        )
        m.add(route.references)

    m.add_ports(g.ports)
    m << gf.components.rectangle(size=size, layer=LAYER.FLOORPLAN)
    m.name = "EBeam_JoaquinMatres_Simon_1"
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    m, _ = test_mask2()
    m.show()
