

Here are the components available in the PDK


Components
=============================


add_fiber_array
----------------------------------------------------

.. autofunction:: ubcpdk.components.add_fiber_array

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.add_fiber_array(gc_port_name='o1', with_loopback=False, optical_routing_type=0, fanout_length=0.0)
  c.plot()



crossing
----------------------------------------------------

.. autofunction:: ubcpdk.components.crossing

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.crossing()
  c.plot()



dbr
----------------------------------------------------

.. autofunction:: ubcpdk.components.dbr

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.dbr(w0=0.5, dw=0.1, n=600, l1=0.07940573770491803, l2=0.07940573770491803)
  c.plot()



dbr_cavity
----------------------------------------------------

.. autofunction:: ubcpdk.components.dbr_cavity

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.dbr_cavity()
  c.plot()



dc_adiabatic
----------------------------------------------------

.. autofunction:: ubcpdk.components.dc_adiabatic

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.dc_adiabatic()
  c.plot()



dc_broadband_te
----------------------------------------------------

.. autofunction:: ubcpdk.components.dc_broadband_te

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.dc_broadband_te()
  c.plot()



dc_broadband_tm
----------------------------------------------------

.. autofunction:: ubcpdk.components.dc_broadband_tm

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.dc_broadband_tm()
  c.plot()



gc_te1310
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1310

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.gc_te1310()
  c.plot()



gc_te1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.gc_te1550()
  c.plot()



gc_te1550_broadband
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_te1550_broadband

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.gc_te1550_broadband()
  c.plot()



gc_tm1550
----------------------------------------------------

.. autofunction:: ubcpdk.components.gc_tm1550

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.gc_tm1550()
  c.plot()



ring_with_crossing
----------------------------------------------------

.. autofunction:: ubcpdk.components.ring_with_crossing

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.ring_with_crossing()
  c.plot()



spiral
----------------------------------------------------

.. autofunction:: ubcpdk.components.spiral

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.spiral(N=6, x_inner_length_cutback=300.0, x_inner_offset=0.0, y_straight_inner_top=0.0, xspacing=3.0, yspacing=3.0)
  c.plot()



straight
----------------------------------------------------

.. autofunction:: ubcpdk.components.straight

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.straight(length=10.0, width=0.5, layer=(1, 0), with_pins=True)
  c.plot()



y_adiabatic
----------------------------------------------------

.. autofunction:: ubcpdk.components.y_adiabatic

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.y_adiabatic()
  c.plot()



y_splitter
----------------------------------------------------

.. autofunction:: ubcpdk.components.y_splitter

.. plot::
  :include-source:

  import ubcpdk

  c = ubcpdk.components.y_splitter()
  c.plot()
