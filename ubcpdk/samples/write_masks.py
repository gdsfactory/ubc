"""Write all mask for the course."""

import ubcpdk.samples.ubc_joaquin_matres1 as m1


def write_m1():
    for mask in [
        m1.test_mask1,
        m1.test_mask2,
        m1.test_mask3,
        m1.test_mask4,
        m1.test_mask5,
    ]:
        m, tm = mask()


if __name__ == "__main__":
    write_m1()
