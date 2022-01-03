from omegaconf import OmegaConf

from ubc import db
from ubc.config import PATH


if __name__ == "__main__":
    for mask_number in range(1, 4):
        mask = PATH.mask / f"EBeam_JoaquinMatres_{mask_number}.tp.yml"
        components = OmegaConf.load(mask)

        reticle = db.Reticle(name="ubc1")
        foundry = db.Foundry(name="ANT")
        foundry_process = db.FoundryProcess(name="passives", foundry=foundry)
        lot = db.Lot(name="lot1", foundry_process=foundry_process, reticle=reticle)
        wafer = db.Wafer(name="wafer1", lot=lot)
        die = db.Die(name="die1", wafer=wafer)

        reticle.insert()
        foundry.insert()
        foundry_process.insert()
        lot.insert()
        wafer.insert()
        die.insert()

        for component in components.values():
            settings = str(component.full) if hasattr(component, "full") else ""
            component_model = db.Component(
                name=component.name,
                settings=settings,
                die=die,
            )
            component_model.insert()

            instance = db.Instance(
                die=die,
                component=component_model,
                x=component.label.x,
                y=component.label.y,
            )
