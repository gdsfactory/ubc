"""Sample mask for the edx course Q1 2023."""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec

import ubcpdk
import ubcpdk.components as pdk
from ubcpdk.tech import LAYER
from ubcpdk.samples.write_mask import write_mask_gds_with_metadata


size = (605, 410)
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
    gf.add_labels.add_labels_to_ports_optical(m)
    m.name = "EBeam_JoaquinMatres_Simon_1"
    return write_mask_gds_with_metadata(m)


if __name__ == "__main__":
    # m = test_mask3()
    # m.write_gds_with_metadata()

    # m1, tm1 = test_mask1()
    # m2, tm2 = test_mask2()
    # m3, tm3 = test_mask3()
    # m = gf.grid([m1, m2, m3])
    m, _ = test_mask1()
    m.show()

    # c = gf.components.disk(wrap_angle_deg=10.0)
    # c.show()

    # c = add_gc(ubcpdk.components.dc_broadband_te())
    # print(c.to_yaml(with_cells=True, with_ports=True))
    # c.write_gds_with_metadata()
