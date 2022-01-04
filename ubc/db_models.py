"""Models

Reticle: patterns that defines the wafer (the mask)

Lot
Wafer
die:

"""

from typing import Optional
import uuid

from pydbantic import DataBaseModel, PrimaryKey


def stringify_uuid():
    return str(uuid.uuid4())


# Software only
class Reticle(DataBaseModel):
    """Reticle refers to the fabricated mask set.
    Contains all instances that will be replicated across the die
    We name them by Fab together with an incremental name (UBC1, UBC2 ...)
    """

    name: str = PrimaryKey()


class DOE(DataBaseModel):
    """Design of experiment"""

    name: str
    reticle: Optional[Reticle]
    id: str = PrimaryKey(default=stringify_uuid)


# Physical
class Foundry(DataBaseModel):
    name: str = PrimaryKey()


class FoundryProcess(DataBaseModel):
    name: str = PrimaryKey()
    foundry: Foundry


class Lot(DataBaseModel):
    """A lot is a group of wafers that go through the same foundry process"""

    name: str = PrimaryKey()
    foundry_process: FoundryProcess
    reticle: Reticle


class Wafer(DataBaseModel):
    name: str = PrimaryKey()
    lot: Lot


class Die(DataBaseModel):
    name: str
    wafer: Wafer
    id: str = PrimaryKey(default=stringify_uuid)


class Component(DataBaseModel):
    """Component it's only defined in gdsfactory software"""

    name: str
    die: Die
    settings: str
    # settings: Dict[str, Union[float, int, str]]
    id: str = PrimaryKey(default=stringify_uuid)


class Instance(DataBaseModel):
    """Instantiation of a Component with X and Y location in the die"""

    die: Die
    component: Component
    x_nm: int
    y_nm: int
    doe: Optional[DOE]
    id: str = PrimaryKey(default=stringify_uuid)


# Data
class Measurement(DataBaseModel):
    name: str
    instance: Instance
    id: str = PrimaryKey(default=stringify_uuid)


class Simulation(DataBaseModel):
    name: str
    component: Component
    id: str = PrimaryKey(default=stringify_uuid)
