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
# # Data analysis MZI
#
# We analyze the following MZI samples from the edx course
#
# MZI1:
# dL_wg=0
#
# MZI2:
# r=5
# dL_path = (208.40000 - 148.15000) * 2
# dL_wg = dL_path + 2*pi*r - 4*2*r
# = 111.915
#
# MZI3:
# r=5
# dL_path = (259.55000-148.15000) * 2
# dL_wg = dL_path + 2*pi*r - 4*2*r ; dL_wg
# = 214.215
#
# MZI4:
# r1 = 435.90000-427.60000; r1
# r2 = 10
# dL_path = (259.55000-148.15000) * 2
# dL_wg = dL_path + pi*(r1+r2) - 4*(r1+r2) ; dL_wg
# = 207.08945
#
# MZI5:
# r1 = 556.35000-547.60000; r1
# r2 = 10
# dL_path = (259.55000-148.15000) * 2
# dL_wg = dL_path + pi*(r1+r2) - 4*(r1+r2) ; dL_wg
# = 206.703125
#
# MZI6:
# r=4
# dL_path = (259.55000-148.15000) * 2
# dL_wg = dL_path + 2*pi*r - 4*2*r ; dL_wg
# = 215.932
#
# MZI8:
# r=3
# dL_path = (259.55000-148.15000) * 2
# dL_wg = dL_path + 2*pi*r - 4*2*r ; dL_wg
# = 217.649
#
# MZI17:
# r=2
# dL_path = (259.55000-148.15000) * 2
# dL_wg = dL_path + 2*pi*r - 4*2*r ; dL_wg
# = 219.366

# %%
import matplotlib.pyplot as plt
import numpy as np

import ubcpdk
from ubcpdk.simulation.circuits.mzi_spectrum import mzi_spectrum

# %%
w, p = ubcpdk.data.read_mat(ubcpdk.PATH.mzi1, port=0)
plt.plot(w, p)

# %% [markdown]
# For some reason this MZI has an interference pattern. This is strange because the lengths of both arms are the same. This means that there was a strong height variation on the chip.

# %%
w, p = ubcpdk.data.read_mat(ubcpdk.PATH.mzi3, port=0)
plt.plot(w, p)

# %%
wr = np.linspace(1520, 1580, 1200) * 1e-3
pr = mzi_spectrum(L1_um=0, L2_um=214.215, wavelength_um=wr)
plt.plot(wr * 1e3, 10 * np.log10(pr))

# %%
w, p = ubcpdk.data.read_mat(ubcpdk.PATH.mzi3, port=0)
pb = ubcpdk.data.remove_baseline(w, p)
plt.plot(w, pb)

# %%
plt.plot(w, pb, label="measurement")
plt.plot(wr * 1e3, 10 * np.log10(pr), label="analytical")
plt.legend()

# %%
# ms.sweep_wavelength?

# %%
from scipy.optimize import curve_fit

L1_um = 40
L2_um = L1_um + 215.932


def mzi_logscale(wavelength_um, alpha, n1, n2, n3):
    return 10 * np.log10(
        mzi_spectrum(
            L1_um=L1_um,
            L2_um=L2_um,
            wavelength_um=wavelength_um,
            alpha=alpha,
            n1=n1,
            n2=n2,
            n3=n3,
        )
    )


w, p = ubcpdk.data.read_mat(ubcpdk.PATH.mzi6, port=0)
wum = w * 1e-3
pb = ubcpdk.data.remove_baseline(w, p)

p0 = [1e-3, 2.4, -1, 0]
plt.plot(w, pb, label="data")
plt.plot(w, mzi_logscale(wum, *p0), label="initial condition")
plt.legend()

# %%
params, params_covariance = curve_fit(mzi_logscale, wum, pb, p0=[1e-3, 2.4, -1, 0])

# %%
params

# %%
plt.plot(w, pb, label="data")
plt.plot(w, mzi_logscale(wum, *params), label="fit")
plt.legend()

# %%
L1_um = 40
L2_um = L1_um + 215.932


def mzi(wavelength_um, alpha, n1, n2, n3):
    return mzi_spectrum(
        L1_um=L1_um,
        L2_um=L2_um,
        wavelength_um=wavelength_um,
        alpha=alpha,
        n1=n1,
        n2=n2,
        n3=n3,
    )


w, p = ubcpdk.data.read_mat(ubcpdk.PATH.mzi6, port=0)
wum = w * 1e-3
pb = ubcpdk.data.remove_baseline(w, p)
pb_linear = 10 ** (pb / 10)

p0 = [1e-3, 2.4, -1, 0]
plt.plot(w, pb_linear, label="data")
plt.plot(w, mzi(wum, *p0), label="initial condition")
plt.legend()

# %%
params, params_covariance = curve_fit(mzi, wum, pb, p0=p0)

# %%
plt.plot(w, pb_linear, label="data")
plt.plot(w, mzi(wum, *params), label="fit")
plt.legend()
