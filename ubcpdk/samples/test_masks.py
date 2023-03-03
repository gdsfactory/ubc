"""Write all mask for the course."""
import shutil

from ubcpdk.config import PATH
import ubcpdk.samples.ubc_joaquin_matres1 as m11
import ubcpdk.samples.ubc_helge as m12
import ubcpdk.samples.ubc_simon as m13


def test_masks_2023_v1():
    """Write all masks for 2023_v1."""
    dirpath = PATH.mask
    dirpath_gds = dirpath / "gds"

    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath_gds.mkdir(exist_ok=True, parents=True)

    for mask in [
        m11.test_mask1,
        m11.test_mask2,
        m11.test_mask3,
        m11.test_mask4,
        m11.test_mask5,
        m11.test_mask6,
        m11.test_mask7,
        m12.test_mask1,
        m12.test_mask2,
        m13.test_mask1,
        m13.test_mask3,
        m13.test_mask4,
        m13.test_mask5,
    ]:
        m, tm = mask()

    for gdspath in dirpath.glob("*.gds"):
        shutil.copyfile(gdspath, dirpath_gds / f"{gdspath.name}")


if __name__ == "__main__":
    test_masks_2023_v1()
