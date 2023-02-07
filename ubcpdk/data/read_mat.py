from gdsfactory.typings import Tuple, PathType
from scipy.io import loadmat
import numpy as np


def read_mat(filename: PathType, port: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Reads .mat file and returns 2 np.arrays (wavelength, power).

    input: (.mat data download filename, port response)
    outputs parsed data array [wavelength (m), power (dBm)]
    data is assumed to be from automated measurement scanResults or scandata format
    based on SiEPIC_Photonics_Package/core.py
    """
    data = loadmat(filename)

    if "scanResults" in data:
        wavelength = data["scanResults"][0][port][0][:, 0]
        power = data["scanResults"][0][port][0][:, 1]
    elif "scandata" in data:
        wavelength = data["scandata"][0][0][0][:][0]
        power = data["scandata"][0][0][1][:, port]
    elif "wavelength" in data:
        wavelength = data["wavelength"][0][:]
        power = data["power"][:, port][:]

    return [wavelength, power]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import ubcpdk

    w, p = read_mat(ubcpdk.PATH.mzi1)
    plt.plot(w, p)
    plt.show()
