

Here are the components available in the PDK


Cells
=============================


add_fiber_array
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_fiber_array

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.add_fiber_array(gc_port_name='o1', with_loopback=False, optical_routing_type=0, fanout_length=0.0, cross_section='strip', layer_label=(66, 0))
  c.plot()



add_fiber_array_pads_rf
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_fiber_array_pads_rf

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.add_fiber_array_pads_rf(component='ring_single_heater', username='JoaquinMatres')
  c.plot()



add_pads
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_pads

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.add_pads(component='ring_single_heater', username='JoaquinMatres')
  c.plot()



add_pads_dc
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_pads_dc

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.add_pads_dc(component='ring_single_heater', spacing=(0.0, 100.0))
  c.plot()



add_pads_rf
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_pads_rf

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.add_pads_rf(component='ring_single_heater', direction='top', spacing=(0.0, 100.0), layer='MTOP')
  c.plot()



add_pins_bbox_siepic
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_pins_bbox_siepic

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.add_pins_bbox_siepic(port_type='optical', layer_pin=(1, 10), pin_length=0.01, bbox_layer=(68, 0), padding=0, remove_layers=False)
  c.plot()



add_pins_bbox_siepic_remove_layers
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_pins_bbox_siepic_remove_layers

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.add_pins_bbox_siepic_remove_layers(port_type='optical', layer_pin=(1, 10), pin_length=0.01, bbox_layer=(68, 0), padding=0, remove_layers=True)
  c.plot()



add_pins_siepic_metal
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_pins_siepic_metal

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.add_pins_siepic_metal(port_type='placement', layer_pin=(1, 11), pin_length=0.01)
  c.plot()



bend
----------------------------------------------------

.. autofunction:: ubcpdk.components.bend

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.bend(angle=90.0, p=0.5, with_arc_floorplan=True, direction='ccw', with_bbox=True, cross_section='strip')
  c.plot()



bend_euler
----------------------------------------------------

.. autofunction:: ubcpdk.components.bend_euler

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.bend_euler(angle=90.0, p=0.5, with_arc_floorplan=True, npoints=100, direction='ccw', with_bbox=True, cross_section='strip')
  c.plot()



bend_s
----------------------------------------------------

.. autofunction:: ubcpdk.components.bend_s

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.bend_s(size=(11.0, 2.0), npoints=99, cross_section='strip', check_min_radius=False)
  c.plot()



coupler
----------------------------------------------------

.. autofunction:: ubcpdk.components.coupler

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.coupler(gap=0.236, length=20.0, dy=4.0, dx=10.0)
  c.plot()



coupler_ring
----------------------------------------------------

.. autofunction:: ubcpdk.components.coupler_ring

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.coupler_ring(gap=0.2, radius=5.0, length_x=4.0, cross_section='strip', length_extension=3)
  c.plot()



dbg
----------------------------------------------------

.. autofunction:: ubcpdk.components.dbg

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.dbg(w0=0.5, dw=0.1, n=100, l1=0.07940573770491803, l2=0.07940573770491803)
  c.plot()



dbr
----------------------------------------------------

.. autofunction:: ubcpdk.components.dbr

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.dbr(w0=0.5, dw=0.1, n=100, l1=0.07940573770491803, l2=0.07940573770491803)
  c.plot()



dbr_cavity
----------------------------------------------------

.. autofunction:: ubcpdk.components.dbr_cavity

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.dbr_cavity()
  c.plot()



dbr_cavity_te
----------------------------------------------------

.. autofunction:: ubcpdk.components.dbr_cavity_te

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.dbr_cavity_te(component='dbr_cavity')
  c.plot()



ebeam_BondPad
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_BondPad

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_BondPad()
  c.plot()



ebeam_adiabatic_te1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_adiabatic_te1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_adiabatic_te1550()
  c.plot()



ebeam_adiabatic_tm1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_adiabatic_tm1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_adiabatic_tm1550()
  c.plot()



ebeam_bdc_te1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_bdc_te1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_bdc_te1550()
  c.plot()



ebeam_bdc_tm1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_bdc_tm1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_bdc_tm1550()
  c.plot()



ebeam_crossing4
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_crossing4

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_crossing4()
  c.plot()



ebeam_crossing4_2ports
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_crossing4_2ports

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_crossing4_2ports()
  c.plot()



ebeam_dc_halfring_straight
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_dc_halfring_straight

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_dc_halfring_straight(gap=0.2, radius=5.0, length_x=4.0, cross_section='strip', siepic=True, model='ebeam_dc_halfring_straight')
  c.plot()



ebeam_dc_te1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_dc_te1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_dc_te1550(gap=0.236, length=20.0, dy=4.0, dx=10.0, cross_section='strip')
  c.plot()



ebeam_splitter_adiabatic_swg_te1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_splitter_adiabatic_swg_te1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_splitter_adiabatic_swg_te1550()
  c.plot()



ebeam_splitter_swg_assist_te1310
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_splitter_swg_assist_te1310

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_splitter_swg_assist_te1310()
  c.plot()



ebeam_splitter_swg_assist_te1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_splitter_swg_assist_te1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_splitter_swg_assist_te1550()
  c.plot()



ebeam_swg_edgecoupler
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_swg_edgecoupler

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_swg_edgecoupler()
  c.plot()



ebeam_terminator_te1310
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_terminator_te1310

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_terminator_te1310()
  c.plot()



ebeam_terminator_te1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_terminator_te1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_terminator_te1550()
  c.plot()



ebeam_terminator_tm1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_terminator_tm1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_terminator_tm1550()
  c.plot()



ebeam_y_1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_y_1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_y_1550()
  c.plot()



ebeam_y_adiabatic
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_y_adiabatic

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_y_adiabatic()
  c.plot()



ebeam_y_adiabatic_1310
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_y_adiabatic_1310

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_y_adiabatic_1310()
  c.plot()



ebeam_y_adiabatic_tapers
----------------------------------------------------

.. autofunction:: ubcpdk.components.ebeam_y_adiabatic_tapers

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ebeam_y_adiabatic_tapers()
  c.plot()



gc_te1310
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1310

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.gc_te1310()
  c.plot()



gc_te1310_8deg
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1310_8deg

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.gc_te1310_8deg()
  c.plot()



gc_te1310_broadband
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1310_broadband

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.gc_te1310_broadband()
  c.plot()



gc_te1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.gc_te1550()
  c.plot()



gc_te1550_90nmSlab
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1550_90nmSlab

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.gc_te1550_90nmSlab()
  c.plot()



gc_te1550_broadband
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1550_broadband

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.gc_te1550_broadband()
  c.plot()



gc_tm1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_tm1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.gc_tm1550()
  c.plot()



metal_via
----------------------------------------------------

.. autofunction:: ubcpdk.components.metal_via

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.metal_via()
  c.plot()



mmi1x2
----------------------------------------------------

.. autofunction:: ubcpdk.components.mmi1x2

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.mmi1x2(width_taper=1.0, length_taper=10.0, length_mmi=5.5, width_mmi=2.5, gap_mmi=0.25, with_bbox=True)
  c.plot()



mzi
----------------------------------------------------

.. autofunction:: ubcpdk.components.mzi

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.mzi(delta_length=10.0, length_y=2.0, length_x=0.1, with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o2', port_e0_combiner='o3', nbends=2, cross_section='strip', mirror_bot=False, add_optical_ports_arms=False)
  c.plot()



mzi_heater
----------------------------------------------------

.. autofunction:: ubcpdk.components.mzi_heater

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.mzi_heater(delta_length=10.0, length_y=2.0, length_x=200, straight_x_top='straight_heater_metal', with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o2', port_e0_combiner='o3', nbends=2, cross_section='strip', mirror_bot=False, add_optical_ports_arms=False)
  c.plot()



pad_array
----------------------------------------------------

.. autofunction:: ubcpdk.components.pad_array

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.pad_array(spacing=(125, 125), columns=6, rows=1, orientation=270)
  c.plot()



photonic_wirebond_surfacetaper_1310
----------------------------------------------------

.. autofunction:: ubcpdk.components.photonic_wirebond_surfacetaper_1310

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.photonic_wirebond_surfacetaper_1310()
  c.plot()



photonic_wirebond_surfacetaper_1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.photonic_wirebond_surfacetaper_1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.photonic_wirebond_surfacetaper_1550()
  c.plot()



ring_double_heater
----------------------------------------------------

.. autofunction:: ubcpdk.components.ring_double_heater

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ring_double_heater(gap=0.2, radius=10.0, length_x=0.01, length_y=0.01, cross_section_heater='heater_metal', cross_section_waveguide_heater='strip_heater_metal', cross_section='strip', port_orientation=90, via_stack_offset=(0, 0))
  c.plot()



ring_single
----------------------------------------------------

.. autofunction:: ubcpdk.components.ring_single

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ring_single(gap=0.2, radius=10.0, length_x=4.0, length_y=0.6, cross_section='strip')
  c.plot()



ring_single_heater
----------------------------------------------------

.. autofunction:: ubcpdk.components.ring_single_heater

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ring_single_heater(gap=0.2, radius=10.0, length_x=4.0, length_y=0.6, cross_section_waveguide_heater='strip_heater_metal', cross_section='strip', port_orientation=90, via_stack_offset=(0, 0))
  c.plot()



ring_with_crossing
----------------------------------------------------

.. autofunction:: ubcpdk.components.ring_with_crossing

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.ring_with_crossing(gap=0.2, length_x=4, length_y=0, radius=5.0, with_component=True, port_name='o4')
  c.plot()



spiral
----------------------------------------------------

.. autofunction:: ubcpdk.components.spiral

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.spiral(N=6, x_inner_length_cutback=300.0, x_inner_offset=0.0, y_straight_inner_top=0.0, xspacing=3.0, yspacing=3.0, cross_section='strip', with_inner_ports=False, y_straight_outer_offset=0.0, inner_loop_spacing_offset=0.0)
  c.plot()



straight
----------------------------------------------------

.. autofunction:: ubcpdk.components.straight

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.straight(length=10.0, npoints=2, with_bbox=True, cross_section='strip')
  c.plot()



straight_one_pin
----------------------------------------------------

.. autofunction:: ubcpdk.components.straight_one_pin

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.straight_one_pin(length=1)
  c.plot()



taper
----------------------------------------------------

.. autofunction:: ubcpdk.components.taper

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.taper(length=10.0, width1=0.5, with_bbox=True, with_two_ports=True, cross_section='strip')
  c.plot()



terminator_short
----------------------------------------------------

.. autofunction:: ubcpdk.components.terminator_short

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.terminator_short()
  c.plot()



thermal_phase_shifter0
----------------------------------------------------

.. autofunction:: ubcpdk.components.thermal_phase_shifter0

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.thermal_phase_shifter0()
  c.plot()



thermal_phase_shifter1
----------------------------------------------------

.. autofunction:: ubcpdk.components.thermal_phase_shifter1

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.thermal_phase_shifter1()
  c.plot()



thermal_phase_shifter2
----------------------------------------------------

.. autofunction:: ubcpdk.components.thermal_phase_shifter2

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.thermal_phase_shifter2()
  c.plot()



thermal_phase_shifter3
----------------------------------------------------

.. autofunction:: ubcpdk.components.thermal_phase_shifter3

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.thermal_phase_shifter3()
  c.plot()



via_stack_heater_mtop
----------------------------------------------------

.. autofunction:: ubcpdk.components.via_stack_heater_mtop

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.via_stack_heater_mtop(size=(10, 10), layers=((11, 0), (12, 0)), vias=(None, None), correct_size=True)
  c.plot()
