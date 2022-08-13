"""Equations for MZI.

w = 1.55
io/ii = 1/2*(1+cos(beta*dl))
beta = 2*np.pi*n/w

beta*l = np.pi
beta = 2*np.pi*n/w

2*np.pi*n*l/w = np.pi
l = w/2/n
l = np.pi/beta
"""
import numpy as np


def get_pi_length(w: float = 1.55, n: float = 2.4) -> float:
    return w / 2 / n


def get_length(power_per_cent: float = 80, w: float = 1.55, n: float = 2.4) -> float:
    """Returns length for a MZI based variable.

    io/ii = 1/2*(1+cos(beta*dl)) = sqrt(power_per_cent)
    1+cos(beta*dl) = 2* power_per_cent
    cos(beta*dl) = 2* power_per_cent - 1
    beta*dl = np.arcos(2* power_per_cent - 1)
    dl = (np.arcos(2* power_per_cent - 1))/beta

    .. code::
                        L+dl
                       ______
         input      __|      |__  output
                 ===____________==
                         L
    """
    beta = 2 * np.pi * n / w
    return (np.arccos(2 * power_per_cent / 100 - 1)) / beta


if __name__ == "__main__":
    # print(get_pi_length())
    print(get_length() * 1e3)
