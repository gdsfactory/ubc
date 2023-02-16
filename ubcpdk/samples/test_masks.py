"""Write all mask for the course."""
from shutil import copyfile

from ubcpdk.config import PATH
import ubcpdk.samples.ubc_joaquin_matres1 as m11
import ubcpdk.samples.ubc_helge as m12
import ubcpdk.samples.ubc_simon as m13


def test_masks_2023_v1():
    """Write all masks for 2023_v1."""
    for mask in [
        m11.test_mask1,
        m11.test_mask2,
        m11.test_mask3,
        m11.test_mask4,
        m11.test_mask5,
        m12.test_mask1,
        m12.test_mask2,
        m13.test_mask1,
        m13.test_mask3,
        m13.test_mask4,
        m13.test_mask5,
    ]:
        dirpath = PATH.mask
        dirpath_gds = dirpath / "gds"

        dirpath.rmdir()
        m, tm = mask()

        for gdspath in dirpath.glob("*.gds"):
            copyfile(gdspath, dirpath_gds)


if __name__ == "__main__":
    test_masks_2023_v1()
