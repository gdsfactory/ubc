import dataclasses
from typing import Callable, List, Optional, Tuple

import pp
from phidl import device_layout as pd
from phidl.device_layout import Label
from pp.add_labels import get_input_label
from pp.cell import cell
from pp.component import Component
from pp.components.ring_single_dut import ring_single_dut
from pp.pdk import Pdk
from pp.port import Port, auto_rename_ports
from pp.rotate import rotate
from pp.routing.add_fiber_array import add_fiber_array
from pp.tech import Tech
from pp.types import ComponentFactory, ComponentReference
from ubc.config import conf
from ubc.import_gds import import_gds
from ubc.tech import LAYER, TECH_SILICON_C

L = 1.55 / 4 / 2 / 2.44


def get_optical_text(
    port: Port,
    gc: ComponentReference,
    gc_index: Optional[int] = None,
    component_name: Optional[str] = None,
) -> str:
    """Return label for a component port and a grating coupler.

    Args:
        port: component port.
        gc: grating coupler reference.
        component_name: optional component name.
    """
    polarization = gc.get_property("polarization")
    wavelength_nm = gc.get_property("wavelength")

    assert polarization.upper() in [
        "TE",
        "TM",
    ], f"Not valid polarization {polarization.upper()} in [TE, TM]"
    assert (
        isinstance(wavelength_nm, (int, float)) and 1000 < wavelength_nm < 2000
    ), f"{wavelength_nm} is Not valid 1000 < wavelength < 2000"

    if component_name:
        name = component_name
    elif type(port.parent) == pp.Component:
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    name = name.replace("_", "-")
    label = (
        f"opt_in_{polarization.upper()}_{int(wavelength_nm)}_device_"
        + f"{conf.username}_({name})-{gc_index}-{port.name}"
    )
    return label


def get_input_labels_all(
    io_gratings,
    ordered_ports,
    component_name,
    layer_label=LAYER.LABEL,
    gc_port_name: str = "W0",
):
    """Return labels (elements list) for all component ports."""
    elements = []
    for i, g in enumerate(io_gratings):
        label = get_input_label(
            port=ordered_ports[i],
            gc=g,
            gc_index=i,
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
        )
        elements += [label]

    return elements


def get_input_labels(
    io_gratings: List[ComponentReference],
    ordered_ports: List[Port],
    component_name: str,
    layer_label: Tuple[int, int] = LAYER.LABEL,
    gc_port_name: str = "W0",
    port_index: int = 1,
) -> List[Label]:
    """Return labels (elements list) for all component ports."""
    if port_index == -1:
        return get_input_labels_all(
            io_gratings=io_gratings,
            ordered_ports=ordered_ports,
            component_name=component_name,
            layer_label=layer_label,
            gc_port_name=gc_port_name,
        )
    gc = io_gratings[port_index]
    port = ordered_ports[1]

    text = get_optical_text(
        port=port, gc=gc, gc_index=port_index, component_name=component_name
    )
    layer, texttype = pd._parse_layer(layer_label)
    label = pd.Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer,
        texttype=texttype,
    )
    return [label]


@cell
def gc_te1550() -> Component:
    c = import_gds("ebeam_gc_te1550")
    c = rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1550
    return c


@cell
def gc_te1550_broadband():
    c = import_gds("ebeam_gc_te1550_broadband")
    c = rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1550
    auto_rename_ports(c)
    return c


@cell
def gc_te1310():
    c = import_gds("ebeam_gc_te1310")
    c = rotate(component=c, angle=180)
    c.polarization = "te"
    c.wavelength = 1310
    return c


@cell
def gc_tm1550():
    c = import_gds("ebeam_gc_tm1550")
    c.polarization = "tm"
    c.wavelength = 1550
    return c


@dataclasses.dataclass
class PdkSiliconCband(Pdk):
    tech: Tech = TECH_SILICON_C

    def grating_coupler(self) -> Component:
        return gc_te1550()

    def grating_coupler_te1550_broadband(self) -> Component:
        return gc_te1550_broadband()

    def grating_coupler_te1310(self) -> Component:
        return gc_te1310()

    def grating_coupler_tm1550(self) -> Component:
        return gc_tm1550()

    def crossing(self) -> Component:
        """TE waveguide crossing."""
        return import_gds("ebeam_crossing4", rename_ports=True)

    def dc_broadband(self) -> Component:
        """Broadband directional coupler TE1550 50/50 power."""
        return import_gds("ebeam_bdc_te1550")

    def dc_adiabatic(self) -> Component:
        """Adiabatic directional coupler TE1550 50/50 power."""
        return import_gds("ebeam_adiabatic_te1550")

    def y_adiabatic(self) -> Component:
        """Adiabatic Y junction TE1550 50/50 power."""
        return import_gds("ebeam_y_adiabatic")

    def y_splitter(self) -> Component:
        """Y junction TE1550 50/50 power."""
        return import_gds("ebeam_y_1550")

    def ring_with_crossing(self, **kwargs) -> Component:
        return ring_single_dut(component=self.crossing(), **kwargs)

    def dbr(self, w0=0.5, dw=0.1, n=600, l1=L, l2=L) -> Component:
        return pp.c.dbr(
            w1=w0 - dw / 2,
            w2=w0 + dw / 2,
            n=n,
            l1=l1,
            l2=l2,
            waveguide_function=self.waveguide,
        )

    def dbr_cavity(self, **kwargs) -> Component:
        return pp.c.cavity(component=self.dbr(**kwargs))

    def spiral(self, **kwargs):
        return pp.c.spiral_external_io(**kwargs)

    def add_fiber_array(
        self,
        component: Component,
        component_name: None = None,
        gc_port_name: str = "W0",
        get_input_labels_function: Callable = get_input_labels,
        with_align_ports: bool = False,
        optical_routing_type: int = 0,
        fanout_length: float = 0.0,
        grating_coupler: Optional[ComponentFactory] = None,
        bend_factory: Optional[ComponentFactory] = None,
        straight_factory: Optional[ComponentFactory] = None,
        taper_factory: Optional[ComponentFactory] = None,
        route_filter: Optional[ComponentFactory] = None,
        bend_radius: Optional[float] = None,
        auto_taper_to_wide_waveguides: bool = False,
        **kwargs,
    ) -> Component:
        """Returns component with grating couplers and labels on each port.

        Routes all component ports south.
        Can add align_ports loopback reference structure on the edges.

        Args:
            component: to connect
            component_name: for the label
            gc_port_name: grating coupler input port name 'W0'
            get_input_labels_function: function to get input labels for grating couplers
            with_align_ports: True, adds loopback structures
            optical_routing_type: None: autoselection, 0: no extension
            fanout_length: None  # if None, automatic calculation of fanout length
            taper_length: length of the taper
            grating_coupler: grating coupler instance, function or list of functions
            bend_factory: function for bends
            optical_io_spacing: SPACING_GC
            straight_factory: waveguide
            taper_factory: taper function
            route_filter: for waveguides and bends
            bend_radius: for bends
        """

        c = add_fiber_array(
            component=component,
            component_name=component_name,
            route_filter=route_filter or self.get_route_euler,
            grating_coupler=grating_coupler or self.grating_coupler,
            bend_factory=bend_factory or self.bend_euler,
            straight_factory=straight_factory or self.waveguide,
            taper_factory=taper_factory or self.taper,
            gc_port_name=gc_port_name,
            get_input_labels_function=get_input_labels_function,
            with_align_ports=with_align_ports,
            optical_routing_type=optical_routing_type,
            layer_label=self.tech.layer_label,
            fanout_length=fanout_length,
            bend_radius=bend_radius or self.tech.bend_radius,
            tech=self.tech,
            auto_taper_to_wide_waveguides=auto_taper_to_wide_waveguides,
            **kwargs,
        )
        c.rotate(-90)
        return c


PDK = PdkSiliconCband()

if __name__ == "__main__":
    p = PDK
    # c = p.ring_single(length_x=6)
    # c = p.dc_broadband()

    c = p.ring_with_crossing()
    c = p.dbr()  # needs fixing
    c = p.dbr_cavity()
    cc = p.add_fiber_array(c)
    cc.show()
