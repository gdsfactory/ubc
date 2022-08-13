import matplotlib.pyplot as plt
import numpy as np


def find_bandwidth(x: np.ndarray, y: np.ndarray, threshold: float = 3) -> float:
    """Return bandwidth of x above a threshold around max(y).

    Args:
        x: wavelength.
        y: power.
        threshold: default bandwidth point (in y scale units)
            3 for dB, 0.5 if Y is in linear scale from 0 to 1.
    """
    index_max = np.argmax(y)
    ymax = y[index_max]
    ybw = ymax - threshold
    yl = y[:index_max]
    yr = y[index_max:]
    xl = x[:index_max]
    xr = x[index_max:]
    il = max(np.argwhere(np.abs(yl - ybw) < 0.1))
    ir = min(np.argwhere(np.abs(yr - ybw) < 0.1))
    return float(xr[ir] - xl[il])


def plot_bandwidth(x, y, threshold: float = 3) -> None:
    index_max = np.argmax(y)
    ymax = y[index_max]
    ybw = ymax - threshold
    yl = y[:index_max]
    yr = y[index_max:]
    xl = x[:index_max]
    xr = x[index_max:]
    il = max(np.argwhere(np.abs(yl - ybw) < 0.1))
    ir = min(np.argwhere(np.abs(yr - ybw) < 0.1))

    plt.plot(x, y)
    plt.plot(xl[il], yl[il], "o", c="k")
    plt.plot(xr[ir], yr[ir], "o", c="k")
    plt.show()


if __name__ == "__main__":
    from ubcpdk.data.dbr import dbrs
    from ubcpdk.data.chop import chop
    from ubcpdk.data.read_mat import read_mat

    w, p = read_mat(dbrs["1_5"], port=1)
    wc, pc = chop(w, p, ymin=-60)

    print(find_bandwidth(wc, pc) * 1e9)
    # plot_bandwidth(wc, pc)
