"""Write all mask for the course."""

import ubcpdk.samples.ubc_joaquin_matres1 as m1
import ubcpdk.samples.ubc_helge as m2


def test_masks_2023_v1():
    """Write all masks for 2023_v1."""
    for mask in [
        m1.test_mask1,
        m1.test_mask2,
        m1.test_mask3,
        m1.test_mask4,
        m1.test_mask5,
        m2.test_mask_1,
        m2.test_mask_2,
    ]:
        m, tm = mask()


if __name__ == "__main__":
    test_masks_2023_v1()
