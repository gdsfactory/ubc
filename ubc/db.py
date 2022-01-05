"""Database Models
Reticle: patterns that defines the wafer (the mask)
Lot
Wafer
die:
"""

from typing import Optional, List

import asyncio
from pydbantic import Database
from pydbantic import DataBaseModel, PrimaryKey


# Software only
class Reticle(DataBaseModel):
    """Reticle refers to the fabricated mask set.
    Contains all instances that will be replicated across the die
    We name them by Fab together with an incremental name (UBC1, UBC2 ...)
    """

    name: str = PrimaryKey()


class DOE(DataBaseModel):
    """Design of experiment"""

    name: str = PrimaryKey()
    reticle: Optional[Reticle]


class Component(DataBaseModel):
    """Component"""

    name: str = PrimaryKey()
    settings: str


class Instance(DataBaseModel):
    """Instances are references to Components"""

    component: Component
    x_nm: int
    y_nm: int
    doe: Optional[DOE]


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
    name: str = PrimaryKey()
    lot: Lot
    wafer: Wafer


class Device(DataBaseModel):
    """A Device is a physical realization of an instance"""

    die: Die
    instance: Instance


# Data
class Measurement(DataBaseModel):
    name: str = PrimaryKey()
    device: Device
    x: List[float]
    y: List[float]
    port1: str
    port2: str


class Simulation(DataBaseModel):
    name: str = PrimaryKey()
    component: Component
    x: List[float]
    y: List[float]
    port1: str
    port2: str


async def setup_database():
    await Database.create(
        "sqlite:///test.db",
        tables=[
            Reticle,
            DOE,
            Component,
            Instance,
            Foundry,
            FoundryProcess,
            Lot,
            Die,
            Device,
            Measurement,
            Simulation,
        ],
    )


if __name__ == "__main__":
    asyncio.run(setup_database())
