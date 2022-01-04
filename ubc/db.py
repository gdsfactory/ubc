"""Database Models

Reticle: patterns that defines the wafer (the mask)
Lot
Wafer
die:

"""

import asyncio
from pydbantic import Database

import ubc.db_models as db


async def setup_database():
    await Database.create(
        "sqlite:///test.db",
        tables=[
            # db.Reticle,
            # db.DOE,
            # db.Foundry,
            # db.FoundryProcess,
            # db.Lot,
            # db.Wafer,
            # db.Die,
            # db.Component,
            db.Instance,
            db.Measurement,
            db.Simulation,
        ],
    )


if __name__ == "__main__":
    asyncio.run(setup_database())
