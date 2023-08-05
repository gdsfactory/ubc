# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Circuit simulations (MEEP + SAX)
#
# ![](https://i.imgur.com/RSOTDIN.png)
#
# Sparameters are common in RF and photonic simulation.
#
# We are going to simulate a MZI interferometer circuit. For that we need to simulate each of the component Sparameters in MEEP and then use a linear circuit solver to solve the Sparameters for the circuit.
#
# Notice that MEEP only works on MacOs and Linux, so if you are on windows you can use tidy3d (not open source) or use Windows WSL.

# ## MEEP FDTD

# +
import gdsfactory as gf
import gplugins as sim
import gplugins.gmeep as gm

import ubcpdk as pdk

# -

c = pdk.components.ebeam_y_1550()
c

c.ports

# `run=False` only plots the simulations for you to review that is set up **correctly**

df = gm.write_sparameters_meep(c, run=False)

df = gm.write_sparameters_meep(c, run=True)

sim.plot.plot_sparameters(df, keys=["s21m"], with_simpler_input_keys=True)

# ## 2.5D FDTD
#
# For faster simulations you can do an effective mode approximation, to compute the mode of the slab and run a 2D simulation to speed your [simulations](https://www.lumerical.com/learn/whitepapers/lumericals-2-5d-fdtd-propagation-method/)

core_material = sim.get_effective_indices(
    core_material=3.4777,
    clad_materialding=1.444,
    nsubstrate=1.444,
    thickness=0.22,
    wavelength=1.55,
    polarization="te",
)[0]
core_material

df2d = gm.write_sparameters_meep(
    c, resolution=20, is_3d=False, material_name_to_meep=dict(si=core_material)
)

gf.simulation.plot.plot_sparameters(df2d)

sim.plot.plot_sparameters(df2d, keys=["s21m"], with_simpler_input_keys=True)
sim.plot.plot_sparameters(df, keys=["s21m"], with_simpler_input_keys=True)

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

gf.simulation.plot.plot_sparameters(
    df, keys=("s21m",), logscale=False, with_simpler_input_keys=True
)

gf.simulation.plot.plot_sparameters(df, keys=("s11m",), with_simpler_input_keys=True)

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

gm.plot.plot_sparameters(df, keys=("s31m",), with_simpler_input_keys=True)

# ## 3D rendering

# +
from gplugins.add_simulation_markers import add_simulation_markers

import ubcpdk as pdk

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
import gdsfactory as gf
import gplugins.gmeep as gm
import gplugins.sax as gsax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax

import ubcpdk as pdk

# -

y = pdk.components.ebeam_y_1550()


mzi = gf.components.mzi(splitter=y, delta_length=10)
mzi


# +
def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def bend_euler(wl=1.5, length=20.0):
    """ "Let's assume a reduced transmission for the euler bend compared to a straight"""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


# -

sp = gm.write_sparameters_meep(y, run=True)
# ebeam_y_1550 = gsax.read.sdict_from_csv(filepath=df)
ebeam_y_1550 = gsax.read.model_from_npz(sp)

type(sp)

netlist = mzi.get_netlist()
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
plt.plot(1e3 * wl, 10 * np.log10(jnp.abs(S["o1", "o2"]) ** 2))
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()

mzi = gf.components.mzi(splitter=y, delta_length=20)
mzi

netlist = mzi.get_netlist()
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
plt.plot(1e3 * wl, 10 * np.log10(jnp.abs(S["o1", "o2"]) ** 2))
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()
