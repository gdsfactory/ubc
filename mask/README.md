# UBC1

## DFT rules (design for testing)

Layout requirements for automated testing

The following are the layout requirements for automated testing:

Please make sure your grating couplers are:

- facing right (waveguide on the right side)
- on a 127 µm pitch vertically
- Counting from top down, the 1st, 3rd and 4th fibre in the array will be used for the outputs (three detectors are sampled simultaneously), and the 2nd fibre will be input. Many test configurations are possible, such as Fibres 1-2, Fibres 2-3, Fibres 1-2-3, etc. It is also possible to test the same device multiple times by placing labels on multiple ports.

Please place a text label (Layer #10 “text” in layout) on the grating coupler to which you would like the 2nd fibre (input) aligned be to:

- opt*in*<polarization={TE, TM}>_<wvl>\_device_<deviceID>\_<comment>
  - example: opt_in_TE_1550_device_LukasChrostowski_MZI1
- The label format is case sensitive, and cannot have extra "\_" characters.
- The labels MUST BE UNIQUE.

Examples layouts that conform to these test rules are provided in folder “Examples”.

More details are also available in the PDF slides. The GDS Layout used in the examples is Sample.gds. The TE & TM example layout is GC_Alignment_TE_TM.gds.

## DFM Design for manufacturing

Each participant is allocated a rectangle of 605 µm (width) X 410 µm (height). You can use multiple blocks (see below for details).
