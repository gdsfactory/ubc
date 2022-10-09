# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Simulation plugins
#
#
# ![](https://i.imgur.com/RSOTDIN.png)
#
# Sparameters are common in RF and photonic simulation.
#
#
#
# ```bash
#
#          top view
#               ________________________________
#              |                               |
#              | xmargin_left                  | port_extension
#              |<--------->       port_margin ||<-->
#           o2_|___________          _________||_o3
#              |           \        /          |
#              |            \      /           |
#              |             ======            |
#              |            /      \           |
#           o1_|___________/        \__________|_o4
#              |   |                 <-------->|
#              |   |ymargin_bot   xmargin_right|
#              |   |                           |
#              |___|___________________________|
#
#         side view
#               ________________________________
#              |                     |         |
#              |                     |         |
#              |                   zmargin_top |
#              |xmargin_left         |         |
#              |<---> _____         _|___      |
#              |     |     |       |     |     |
#              |     |     |       |     |     |
#              |     |_____|       |_____|     |
#              |       |                       |
#              |       |                       |
#              |       |zmargin_bot            |
#              |       |                       |
#              |_______|_______________________|
#
#
#
# ```
#
# We are going to simulate a MZI interferometer circuit. For that we need to simulate each of the component Sparameters in Meep and then use a linear circuit solver to solve the Sparameters for the circuit.

# ## Mode solver
#

# +
from gdsfactory.simulation.modes import find_modes_waveguide

def silicon_index(wl):
    """ a rudimentary silicon refractive index model """
    a = 0.2411478522088102
    b = 3.3229394315868976
    return a / wl + b

nm = 1e-3
wl = 1.55
w = 500*nm
modes = find_modes_waveguide(wavelength=wl, wg_width=w, mode_number=1, wg_thickness=0.22, slab_thickness=0.0, ncore=silicon_index(wl), nclad=1.4)
# -

mode = modes[1]

mode.plot_e_all()

mode.neff

# ## FDTD

# +
import gdsfactory.simulation.gmeep as gm
import gdsfactory.simulation as sim
import gdsfactory as gf

import ubcpdk as pdk
# -

c = pdk.components.ebeam_y_1550()
c.unlock()
c.auto_rename_ports()
c

c.ports

# `run=False` only plots the simulations for you to review that is set up **correctly**

df = gm.write_sparameters_meep(c, run=False)

df = gm.write_sparameters_meep(c, run=True)

sim.plot.plot_sparameters(df, keys=['s21m'])

# ## 2.5D FDTD
#
# For faster simulations you can do an effective mode approximation, to compute the mode of the slab and run a 2D simulation to speed your [simulations](https://www.lumerical.com/learn/whitepapers/lumericals-2-5d-fdtd-propagation-method/)

ncore = sim.get_effective_indices(
            ncore=3.4777,
            ncladding=1.444,
            nsubstrate=1.444,
            thickness=0.22,
            wavelength=1.55,
            polarization="te",
        )[0]
ncore

df2d = gm.write_sparameters_meep(c, resolution=20, is_3d=False, material_name_to_meep=dict(si=ncore))

gf.simulation.plot.plot_sparameters(df2d)

sim.plot.plot_sparameters(df2d, keys=['s21m'])
sim.plot.plot_sparameters(df, keys=['s21m'])

# For a small taper S21 (Transmission) is around 0dB (100% transmission)

# ## Port symmetries
#
# You can save some simulations in reciprocal devices.
# If the device looks the same going from in -> out as out -> in, we only need to run one simulation

c = gf.components.bend_euler(radius=3)
c

df = gm.write_sparameters_meep_1x1_bend90(c, run=False)

df = gm.write_sparameters_meep_1x1_bend90(c, run=True)

gf.simulation.plot.plot_sparameters(df)

gf.simulation.plot.plot_sparameters(df, keys=("s21m",), logscale=False)

gf.simulation.plot.plot_sparameters(df, keys=("s11m",))

c = pdk.components.ebeam_crossing4(decorator=gf.port.auto_rename_ports)
c

# Here are the port symmetries for a crossing
#
# ```python
# port_symmetries = {
#     "o1": {
#         "s11": ["s22", "s33", "s44"],
#         "s21": ["s12", "s34", "s43"],
#         "s31": ["s13", "s24", "s42"],
#         "s41": ["s14", "s23", "s32"],
#     }
# }
# ```

df = gm.write_sparameters_meep(
    c,
    resolution=20,
    ymargin=0,
    port_symmetries=gm.port_symmetries.port_symmetries_crossing,
    run=False,
)

df = gm.write_sparameters_meep(
    c,
    resolution=20,
    ymargin=0,
    port_symmetries=gm.port_symmetries.port_symmetries_crossing,
    run=True,
)

gm.plot.plot_sparameters(df)

gm.plot.plot_sparameters(df, keys=("s31m",))

# ## Multicore (MPI)
#
# You can divide each simulation into multiple cores thanks to [MPI (message passing interface)](https://en.wikipedia.org/wiki/Message_Passing_Interface)
#
# Lets try to reproduce the coupler results from the [Meep docs](https://meep.readthedocs.io/en/latest/Python_Tutorials/GDSII_Import/)
#
# According to the simulations in the doc to get a 3dB (50%/50%) splitter you need 150nm over 8um

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import gdsfactory as gf
import gdsfactory.simulation as sim
import gdsfactory.simulation.gmeep as gm
# -

c = gf.components.coupler(length=8, gap=0.13)
c

gm.write_sparameters_meep(component=c, run=False)

filepath = gm.write_sparameters_meep_mpi(
    component=c,
    cores=4,
    resolution=30,
)

df = pd.read_csv(filepath)

gf.simulation.plot.plot_sparameters(df)

gf.simulation.plot.plot_sparameters(df, keys=["s13m", "s14m"])

# ## Batch
#
# You can also run a batch of multicore simulations

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gdsfactory as gf

import gdsfactory.simulation as sim
import gdsfactory.simulation.gmeep as gm
# -

c = gf.components.straight(length=3.1)

gm.write_sparameters_meep(c, ymargin=3, run=False)

# +
c1_dict = {"component": c, "ymargin": 3}
jobs = [
    c1_dict,
]

filepaths = gm.write_sparameters_meep_batch_1x1(
    jobs=jobs,
    cores_per_run=4,
    total_cores=8,
    lazy_parallelism=True,
)
# -

df = pd.read_csv(filepaths[0])
gf.simulation.plot.plot_sparameters(df)

c = gf.components.coupler_ring()
c

p = 2.5
gm.write_sparameters_meep(c, ymargin=0, ymargin_bot=p, xmargin=p, run=False)

# +
c1_dict = dict(
    component=c,
    ymargin=0,
    ymargin_bot=p,
    xmargin=p,
)
jobs = [c1_dict]

filepaths = gm.write_sparameters_meep_batch(
    jobs=jobs,
    cores_per_run=4,
    total_cores=8,
    delete_temp_files=False,
    lazy_parallelism=True,
)
# -

df = pd.read_csv(filepaths[0])

gm.plot.plot_sparameters(df)

gm.plot.plot_sparameters(df, keys=["s31m", "s41m"])

gm.plot.plot_sparameters(df, keys=["s31m"])

gm.plot.plot_sparameters(df, keys=["s41m"])

# ## 3D rendering

# +
from gdsfactory.simulation.add_simulation_markers import add_simulation_markers
import ubcpdk as pdk
import sax

y = pdk.components.ebeam_y_1550()
y.unlock()
y.auto_rename_ports()
y = add_simulation_markers(y)
y
# -

scene = y.to_3d()
scene.show()

# ## Circuit simulation
#

y = pdk.components.ebeam_y_1550()
y.unlock()
y.auto_rename_ports()
y

df = gm.write_sparameters_meep(y)

mzi10 = gf.components.mzi(splitter=y, delta_length=10)
mzi10

# +
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf
import sax
from pprint import pprint

import gdsfactory as gf
import gdsfactory.simulation.sax as gsax
import gdsfactory.simulation.gmeep as gm

import ubcpdk as pdk
# -

y = pdk.components.ebeam_y_1550()
y.unlock()
y.auto_rename_ports()
y

mzi = gf.components.mzi(splitter=y, delta_length=10)
mzi


# +
def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    wl0 = 1.5  # center wavelength for which the waveguide model is defined
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def bend_euler(wl=1.5, length=20.0):
    """ "Let's assume a reduced transmission for the euler bend compared to a straight"""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}



# -

df = gm.write_sparameters_meep(y, run=True)
#ebeam_y_1550 = gsax.read.sdict_from_csv(filepath=df)
ebeam_y_1550 = gsax.read.model_from_csv(filepath=df)

netlist = mzi.get_netlist_dict()
circuit, _ = sax.circuit(
    netlist=netlist,
    models={
        "bend_euler": bend_euler,
        "ebeam_y_1550": ebeam_y_1550,
        "straight": straight,
    },
)

wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)
plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, 10*np.log10(jnp.abs(S["o1", "o2"]) ** 2))
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()

mzi = gf.components.mzi(splitter=y, delta_length=20)
mzi

netlist = mzi.get_netlist_dict()
circuit, _ = sax.circuit(
    netlist=netlist,
    models={
        "bend_euler": bend_euler,
        "ebeam_y_1550": ebeam_y_1550,
        "straight": straight,
    },
)

wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)
plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, 10*np.log10(jnp.abs(S["o1", "o2"]) ** 2))
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()


