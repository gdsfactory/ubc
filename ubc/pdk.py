import dataclasses

from pp.pdk import Pdk
from pp.tech import Tech
from ubc.tech import TECH_SILICON_C


@dataclasses.dataclass
class PdkSiliconCband(Pdk):
    tech: Tech = TECH_SILICON_C


PDK = PdkSiliconCband()

if __name__ == "__main__":
    p = PDK
    c = p.ring_single(length_x=6)
    cc = p.add_fiber_array(c)
    cc.show()
