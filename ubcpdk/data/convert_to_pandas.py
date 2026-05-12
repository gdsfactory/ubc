import pandas as pd
from gdsfactory.typings import PathType
from scipy.io import loadmat


def convert_to_pandas(filename: PathType, port: int = 0) -> pd.DataFrame:
    """Reads .mat file and parses it into a pandas DataFrame."""
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

    return pd.DataFrame({"wavelength": wavelength, "output_power": power})


if __name__ == "__main__":
    import json

    from ubcpdk.config import PATH

    for filepath in [
        PATH.ring_te_r3_g100,
        PATH.ring_te_r3_g150,
        PATH.ring_te_r3_g150,
        PATH.ring_te_r10_g50,
        PATH.ring_te_r10_g100,
        PATH.ring_te_r10_g150,
    ]:
        df = convert_to_pandas(filepath)
        json_path = filepath.with_suffix(".json")
        d = df.to_dict(orient="split")
        json_path.write_text(json.dumps(d, indent=2))
