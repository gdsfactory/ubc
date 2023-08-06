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

# # Data analysis ring
#
# We analyze the following ring samples from the edx course
#
# Double-bus ring resonators
# symmetrically coupled
#
# TE:
# R = [3]
# g = [50, 100, 150]
#
# R = [10]
# g = [50, 100, 150, 200]
#
# TM:
# R = [30]
# g = [150, 200, 250]

# + attributes={"classes": [], "id": "", "n": "2"}
import matplotlib.pyplot as plt

import ubcpdk

# + attributes={"classes": [], "id": "", "n": "3"}
w, p = ubcpdk.data.read_mat(ubcpdk.PATH.ring_te_r3_g100, port=0)
plt.plot(w * 1e9, p)
