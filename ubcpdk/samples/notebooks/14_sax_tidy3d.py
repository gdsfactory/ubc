# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SAX circuit simulator
#
# [SAX](https://flaport.github.io/sax/) is a circuit solver written in JAX, writing your component models in SAX enables you not only to get the function values but the gradients, this is useful for circuit optimization.
#
# This tutorial has been adapted from the SAX Quick Start.
#
# You can install sax with pip (read the SAX install instructions [here](https://github.com/flaport/sax#installation))
#
# ```
# pip install 'gplugins[sax]'
# ```

# %%
from pprint import pprint

import gdsfactory as gf
import gplugins.gtidy3d as gt
import gplugins.sax as gs
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax

# %% [markdown]
# ## Scatter *dictionaries*
#
# The core datastructure for specifying scatter parameters in SAX is a dictionary... more specifically a dictionary which maps a port combination (2-tuple) to a scatter parameter (or an array of scatter parameters when considering multiple wavelengths for example). Such a specific dictionary mapping is called ann `SDict` in SAX (`SDict ≈ Dict[Tuple[str,str], float]`).
#
# Dictionaries are in fact much better suited for characterizing S-parameters than, say, (jax-)numpy arrays due to the inherent sparse nature of scatter parameters. Moreover, dictionaries allow for string indexing, which makes them much more pleasant to use in this context.
#
# ```
# o2            o3
#    \        /
#     ========
#    /        \
# o1            o4
# ```

# %%
nm = 1e-3
coupling = 0.5
kappa = coupling**0.5
tau = (1 - coupling) ** 0.5
coupler_dict = {
    ("o1", "o4"): tau,
    ("o4", "o1"): tau,
    ("o1", "o3"): 1j * kappa,
    ("o3", "o1"): 1j * kappa,
    ("o2", "o4"): 1j * kappa,
    ("o4", "o2"): 1j * kappa,
    ("o2", "o3"): tau,
    ("o3", "o2"): tau,
}
coupler_dict

# %% [markdown]
#  it can still be tedious to specify every port in the circuit manually. SAX therefore offers the `reciprocal` function, which auto-fills the reverse connection if the forward connection exist. For example:

# %%
coupler_dict = sax.reciprocal(
    {
        ("o1", "o4"): tau,
        ("o1", "o3"): 1j * kappa,
        ("o2", "o4"): 1j * kappa,
        ("o2", "o3"): tau,
    }
)

coupler_dict

# %% [markdown]
# ## Parametrized Models
#
# Constructing such an `SDict` is easy, however, usually we're more interested in having parametrized models for our components. To parametrize the coupler `SDict`, just wrap it in a function to obtain a SAX `Model`, which is a keyword-only function mapping to an `SDict`:
#


# %%
def coupler(coupling=0.5) -> sax.SDict:
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    return sax.reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


coupler(coupling=0.3)


# %%
def waveguide(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0) -> sax.SDict:
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    transmission = 10 ** (-loss * length / 20) * jnp.exp(1j * phase)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


# %% [markdown]
# ### Waveguide model
#
# You can create a dispersive waveguide model in SAX.

# %% [markdown]
# Lets compute the effective index `neff` and group index `ng` for a 1550nm 500nm straight waveguide

# %%
strip = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=0.5,
    core_thickness=0.22,
    slab_thickness=0.0,
    core_material="si",
    clad_material="sio2",
    group_index_step=10 * nm,
)
strip.plot_field(field_name="Ex", mode_index=0)  # TE


# %%
neff = strip.n_eff[0]
print(neff)

# %%
ng = strip.n_group[0]
print(ng)

# %%
straight_sc = gf.partial(gs.models.straight, neff=neff, ng=ng)

# %%
gs.plot_model(straight_sc)
plt.ylim(-1, 1)

# %%
gs.plot_model(straight_sc, phase=True)

# %% [markdown]
# ### Coupler model
#
# Lets define the model for an evanescent coupler

# %%
c = gf.components.coupler(length=10, gap=0.2)
c.plot()

# %%
nm = 1e-3
cp = gt.modes.WaveguideCoupler(
    wavelength=1.55,
    core_width=(500 * nm, 500 * nm),
    gap=200 * nm,
    core_thickness=220 * nm,
    slab_thickness=0 * nm,
    core_material="si",
    clad_material="sio2",
)

cp.plot_field(field_name="Ex", mode_index=0)  # even mode
cp.plot_field(field_name="Ex", mode_index=1)  # odd mode


# %% [markdown]
# For a 200nm gap the effective index difference `dn` is `0.026`, which means that there is 100% power coupling over 29.4

# %% [markdown]
# If we ignore the coupling from the bend `coupling0 = 0` we know that for a 3dB coupling we need half of the `lc` length, which is the length needed to coupler `100%` of power.

# %%
coupler_sc = gf.partial(gs.models.coupler, dn=0.026, length=29.4 / 2, coupling0=0)
gs.plot_model(coupler_sc)

# %% [markdown]
# ## SAX gdsfactory Compatibility
# > From Layout to Circuit Model
#
# If you define your SAX S parameter models for your components, you can directly simulate your circuits from gdsfactory

# %%
mzi = gf.components.mzi(delta_length=10)
mzi.plot()

# %%
netlist = mzi.get_netlist()
pprint(netlist["connections"])


# %% [markdown]
# The netlist has three different components:
#
# 1. straight
# 2. mmi1x2
# 3. bend_euler
#
# You need models for each subcomponents to simulate the Component.


# %%
def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def mmi1x2():
    """Assumes a perfect 1x2 splitter"""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def bend_euler(wl=1.5, length=20.0):
    """ "Let's assume a reduced transmission for the euler bend compared to a straight"""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
}

# %%
circuit, _ = sax.circuit(netlist=netlist, models=models)

# %%
circuit, _ = sax.circuit(netlist=netlist, models=models)
wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()

# %%
mzi = gf.components.mzi(delta_length=20)  # Double the length, reduces FSR by 1/2
mzi.plot()

# %%
circuit, _ = sax.circuit(netlist=mzi.get_netlist(), models=models)

wl = np.linspace(1.5, 1.6, 256)
S = circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()

# %% [markdown]
# ## Heater model
#
# You can make a phase shifter model that depends on the applied volage. For that you need first to figure out what's the model associated to your phase shifter, and what is the parameter that you need to tune.

# %%
delta_length = 10
mzi_component = gf.components.mzi_phase_shifter_top_heater_metal(
    delta_length=delta_length
)
mzi_component.plot()


# %%
def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def mmi1x2() -> sax.SDict:
    """Returns a perfect 1x2 splitter."""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def bend_euler(wl=1.5, length=20.0) -> sax.SDict:
    """Returns bend Sparameters with reduced transmission compared to a straight."""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


def phase_shifter_heater(
    wl: float = 1.55,
    neff: float = 2.34,
    voltage: float = 0,
    length: float = 10,
    loss: float = 0.0,
) -> sax.SDict:
    """Returns simple phase shifter model"""
    deltaphi = voltage * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
    "straight_heater_metal_undercut": phase_shifter_heater,
}

# %%
mzi_component = gf.components.mzi_phase_shifter_top_heater_metal(
    delta_length=delta_length
)
netlist = mzi_component.get_netlist()
mzi_circuit, _ = sax.circuit(netlist=netlist, models=models)
S = mzi_circuit(wl=1.55)
S

# %%
wl = np.linspace(1.5, 1.6, 256)
S = mzi_circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()

# %% [markdown]
# Now you can tune the phase shift applied to one of the arms.
#
# How do you find out what's the name of the netlist component that you want to tune?
#
# You can backannotate the netlist and read the labels on the backannotated netlist or you can plot the netlist

# %%
mzi_component.plot_netlist()

# %% [markdown]
# As you can see the top phase shifter instance `sxt` is hard to see on the netlist.
# You can also reconstruct the component using the netlist and look at the labels in klayout.

# %%
mzi_yaml = mzi_component.get_netlist_yaml()
mzi_component2 = gf.read.from_yaml(mzi_yaml)
fig = mzi_component2.plot(label_aliases=True)

# %% [markdown]
# The best way to get a deterministic name of the `instance` is naming the reference on your Pcell.

# %%
voltages = np.linspace(-1, 1, num=5)
voltages = [-0.5, 0, 0.5]

for voltage in voltages:
    S = mzi_circuit(
        wl=wl,
        sxt={"voltage": voltage},
    )
    plt.plot(wl * 1e3, abs(S["o1", "o2"]) ** 2, label=f"{voltage}V")
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

plt.title("MZI vs voltage")
plt.legend()

# %% [markdown]
# ## Variable splitter
#
# You can build a variable splitter by adding a delta length between two 50% power splitters
#
# ![](https://i.imgur.com/xoyIGLn.png)
#
# For example adding a 60um delta length you can build a 90% power splitter


# %%
@gf.cell
def variable_splitter(delta_length: float, splitter=gf.c.mmi2x2):
    return gf.c.mzi2x2_2x2(splitter=splitter, delta_length=delta_length)


nm = 1e-3
c = variable_splitter(delta_length=60 * nm, cache=False)
c.plot()

# %%
models = {
    "bend_euler": gs.models.bend,
    "mmi2x2": gs.models.mmi2x2,
    "straight": gs.models.straight,
}

netlist = c.get_netlist()
circuit, _ = sax.circuit(netlist=netlist, models=models)
wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o3"]) ** 2, label="T")
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()

# %% [markdown]
# ## Coupler sim
#
# Lets compare one coupler versus two coupler

# %%
c = gf.components.coupler(length=29.4, gap=0.2)
c.plot()

# %%
coupler50 = gf.partial(gs.models.coupler, dn=0.026, length=29.4 / 2, coupling0=0)
gs.plot_model(coupler50)


# %% [markdown]
# As you can see the 50% coupling is only at one wavelength (1550nm)
#
# You can chain two couplers to increase the wavelength range for 50% operation.


# %%
@gf.cell
def broadband_coupler(delta_length=0, splitter=gf.c.coupler):
    return gf.c.mzi2x2_2x2(
        splitter=splitter, combiner=splitter, delta_length=delta_length
    )


c = broadband_coupler(delta_length=120 * nm, cache=False)
c.plot()

# %%
c = broadband_coupler(delta_length=164 * nm, cache=False)
models = {
    "bend_euler": gs.models.bend,
    "coupler": coupler50,
    "straight": gs.models.straight,
}

netlist = c.get_netlist()
circuit, _ = sax.circuit(netlist=netlist, models=models)
wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
# plt.plot(1e3 * wl, jnp.abs(S["o1", "o3"]) ** 2, label='T')
plt.plot(1e3 * wl, 20 * np.log10(jnp.abs(S["o1", "o3"])), label="T")
plt.plot(1e3 * wl, 20 * np.log10(jnp.abs(S["o1", "o4"])), label="K")
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.legend()
plt.grid(True)

# %% [markdown]
# As you can see two couplers have more broadband response
