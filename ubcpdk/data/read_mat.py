from scipy.io import loadmat


def read_mat(filename, port=0):
    """reads .mat file and returns np.array
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
