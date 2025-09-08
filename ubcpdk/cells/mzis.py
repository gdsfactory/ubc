"""Mach-Zehnder Interferometers (MZIs) are a type of interferometer used in optics.

They are used to measure the phase shift between two beams of light.

MZIs are used in a variety of applications, including optical communications, quantum computing, and sensing.
"""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def mzi_1x1(
    delta_length: float = 10,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    splitter: ComponentSpec = "ebeam_y_1550",
    combiner: ComponentSpec | None = None,
    port_e1_splitter: str = "o2",
    port_e0_splitter: str = "o3",
    port_e1_combiner: str = "o2",
    port_e0_combiner: str = "o3",
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Mzi.

    Args:
        delta_length: bottom arm vertical extra length.
        bend: 90 degrees bend library.
        straight: straight function.
        splitter: splitter function.
        combiner: combiner function.
        port_e1_splitter: east top splitter port.
        port_e0_splitter: east bot splitter port.
        port_e1_combiner: east top combiner port.
        port_e0_combiner: east bot combiner port.
        cross_section: for routing (sxtop/sxbot to combiner).
    """
    combiner = combiner or splitter
    _splitter = gf.get_component(splitter)
    _combiner = gf.get_component(combiner)

    return gf.c.mzi(
        delta_length=delta_length,
        bend=bend,
        straight=straight,
        splitter=splitter,
        combiner=combiner,
        port_e1_splitter=port_e1_splitter,
        port_e0_splitter=port_e0_splitter,
        port_e1_combiner=port_e1_combiner,
        port_e0_combiner=port_e0_combiner,
        cross_section=cross_section,
        length_y=2.0,
        length_x=0.1,
        straight_y=None,
        straight_x_top=None,
        straight_x_bot=None,
        with_splitter=True,
        port1="o1",
        port2="o2",
        nbends=2,
        cross_section_x_top=None,
        cross_section_x_bot=None,
        mirror_bot=False,
        add_optical_ports_arms=False,
        min_length=0.01,
        auto_rename_ports=True,
    )


@gf.cell
def mzi(
    delta_length: float = 10,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    splitter: ComponentSpec = "coupler",
    combiner: ComponentSpec | None = None,
    port_e1_splitter: str = "o3",
    port_e0_splitter: str = "o4",
    port_e1_combiner: str = "o3",
    port_e0_combiner: str = "o4",
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Mzi.

    Args:
        delta_length: bottom arm vertical extra length.
        bend: 90 degrees bend library.
        straight: straight function.
        splitter: splitter function.
        combiner: combiner function.
        port_e1_splitter: east top splitter port.
        port_e0_splitter: east bot splitter port.
        port_e1_combiner: east top combiner port.
        port_e0_combiner: east bot combiner port.
        cross_section: for routing (sxtop/sxbot to combiner).
    """
    combiner = combiner or splitter
    _splitter = gf.get_component(splitter)
    _combiner = gf.get_component(combiner)
    if len(_splitter.ports) < 4:
        raise ValueError(
            f"Splitter {splitter} has {len(_splitter.ports)} ports, but needs at least 4 ports."
        )
    if len(_combiner.ports) < 4:
        raise ValueError(
            f"Combiner {combiner} has {len(_combiner.ports)} ports, but needs at least 4 ports."
        )

    return gf.c.mzi(
        delta_length=delta_length,
        bend=bend,
        straight=straight,
        splitter=splitter,
        combiner=combiner,
        port_e1_splitter=port_e1_splitter,
        port_e0_splitter=port_e0_splitter,
        port_e1_combiner=port_e1_combiner,
        port_e0_combiner=port_e0_combiner,
        cross_section=cross_section,
        length_y=2.0,
        length_x=0.1,
        straight_y=None,
        straight_x_top=None,
        straight_x_bot=None,
        with_splitter=True,
        port1="o1",
        port2="o2",
        nbends=2,
        cross_section_x_top=None,
        cross_section_x_bot=None,
        mirror_bot=False,
        add_optical_ports_arms=False,
        min_length=0.01,
        auto_rename_ports=True,
    )


@gf.cell
def mzi_heater(
    delta_length: float = 10,
    length_x: float = 200,
    splitter: ComponentSpec = "ebeam_y_1550",
    straight_x_top: ComponentSpec = "straight_heater_metal",
    **kwargs,
) -> gf.Component:
    """Mzi with heater on the bottom arm.

    Args:
        length_x: horizontal length of the top arm (with heater).
        delta_length: bottom arm vertical extra length.
        splitter: splitter function.
        straight_x_top: straight function for the top arm (with heater).
        kwargs: other arguments for mzi.
    """
    c = gf.c.mzi_phase_shifter(
        length_x=length_x,
        delta_length=delta_length,
        splitter=splitter,
        straight_x_top=straight_x_top,
        **kwargs,
    )

    return c


@gf.cell
def mzi_heater_2x2(
    delta_length: float = 10,
    length_x: float = 200,
    splitter: ComponentSpec = "mmi2x2",
    combiner: ComponentSpec = "mmi2x2",
    straight_x_top: ComponentSpec = "straight_heater_metal",
    **kwargs,
) -> gf.Component:
    """Mzi 2x2 with heater on the top arm.

    Args:
        length_x: horizontal length of the top arm (with heater).
        delta_length: bottom arm vertical extra length.
        splitter: splitter function.
        combiner: combiner function.
        straight_x_top: straight function for the top arm (with heater).
        kwargs: other arguments for mzi.
    """
    c = gf.c.mzi2x2_2x2(
        length_x=length_x,
        delta_length=delta_length,
        splitter=splitter,
        combiner=combiner,
        straight_x_top=straight_x_top,
        **kwargs,
    )

    return c


if __name__ == "__main__":
    from ubcpdk import PDK

    PDK.activate()
    c = mzi_heater_2x2()
    c.show()
