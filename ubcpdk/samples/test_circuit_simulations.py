import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax

import ubcpdk
from ubcpdk import PDK


def test_mzi():
    c = ubcpdk.components.mzi(delta_length=20)
    netlist = c.get_netlist()
    models = PDK.models
    circuit, _ = sax.circuit(netlist, models=models)  # type: ignore
    wl = jnp.linspace(1.5, 1.6)

    S = circuit(wl=wl)
    assert S


if __name__ == "__main__":
    c = ubcpdk.components.mzi(delta_length=20)
    netlist = c.get_netlist()
    models = PDK.models
    circuit, _ = sax.circuit(netlist, models=models)  # type: ignore
    wl = jnp.linspace(1.5, 1.6)

    S = circuit(wl=wl)
    plt.figure(figsize=(14, 4))
    plt.title("MZI")
    plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)  # type: ignore
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.grid(True)
    plt.show()
