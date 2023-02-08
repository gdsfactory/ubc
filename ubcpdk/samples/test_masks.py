"""Write all mask for the course."""

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
    ]:
        m, tm = mask()


if __name__ == "__main__":
    test_masks_2023_v1()
