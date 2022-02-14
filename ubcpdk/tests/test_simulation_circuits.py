import numpy as np
import pytest

from gdsfactory.simulation.simphony.get_transmission import get_transmission
from ubcpdk.simulation.circuits_simphony import model_factory, circuit_factory

model_names = model_factory.keys()
circuit_names = circuit_factory.keys()


@pytest.mark.parametrize("model_name", model_names)
def test_elements(model_name, data_regression):
    c = model_factory[model_name]()
    wav = np.linspace(1520, 1570, 3) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)
    _, rows, cols = np.shape(s)
    sdict = {
        f"s{i+1}{j+1}": np.round(np.abs(s[:, i, j]), decimals=3).tolist()
        for i in range(rows)
        for j in range(cols)
    }
    data_regression.check(sdict)


@pytest.mark.parametrize("circuit_name", circuit_names)
def test_circuits(circuit_name, data_regression):
    c = circuit_factory[circuit_name]()
    r = get_transmission(c, num=3)
    s = np.round(r["s"], decimals=3).tolist()
    data_regression.check(dict(w=r["wavelengths"].tolist(), s=s))
