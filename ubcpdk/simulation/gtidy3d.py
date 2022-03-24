"""Customize tidy3d simulations"""
from functools import partial
import gdsfactory.simulation.gtidy3d as gt
from gdsfactory.simulation.gtidy3d.get_simulation import plot_simulation
from gdsfactory.simulation.gtidy3d import materials, utils
from gdsfactory.simulation.gtidy3d.materials import MATERIAL_NAME_TO_TIDY3D_INDEX

from ubcpdk.config import PATH
from ubcpdk.tech import LAYER_STACK

material_name_to_tidy3d_index = MATERIAL_NAME_TO_TIDY3D_INDEX


write_sparameters = partial(
    gt.write_sparameters,
    layer_stack=LAYER_STACK,
    dirpath=PATH.sparameters,
    material_name_to_tidy3d_index=material_name_to_tidy3d_index,
)

write_sparameters_batch = partial(
    gt.write_sparameters_batch,
    layer_stack=LAYER_STACK,
    dirpath=PATH.sparameters,
    material_name_to_tidy3d_index=material_name_to_tidy3d_index,
)

write_sparameters_batch_1x1 = partial(
    gt.write_sparameters_batch_1x1,
    layer_stack=LAYER_STACK,
    dirpath=PATH.sparameters,
    material_name_to_tidy3d_index=material_name_to_tidy3d_index,
)

write_sparameters_1x1 = partial(
    gt.write_sparameters_1x1,
    layer_stack=LAYER_STACK,
    dirpath=PATH.sparameters,
    material_name_to_tidy3d_index=material_name_to_tidy3d_index,
)

get_simulation = partial(
    gt.get_simulation,
    layer_stack=LAYER_STACK,
    material_name_to_tidy3d_index=material_name_to_tidy3d_index,
)

get_simulation_grating_coupler = partial(
    gt.get_simulation_grating_coupler,
    layer_stack=LAYER_STACK,
    material_name_to_tidy3d_index=material_name_to_tidy3d_index,
    fiber_port_name="o1",
)

write_sparameters_grating_coupler = partial(
    gt.write_sparameters_grating_coupler,
    layer_stack=LAYER_STACK,
    material_name_to_tidy3d_index=material_name_to_tidy3d_index,
    dirpath=PATH.sparameters,
    fiber_port_name="o1",
)

write_sparameters_grating_coupler_batch = partial(
    gt.write_sparameters_grating_coupler_batch,
    layer_stack=LAYER_STACK,
    material_name_to_tidy3d_index=material_name_to_tidy3d_index,
    dirpath=PATH.sparameters,
    fiber_port_name="o1",
)

__all__ = [
    "plot_simulation",
    "materials",
    "utils",
    "write_sparameters",
    "write_sparameters_1x1",
    "write_sparameters_batch",
    "write_sparameters_batch_1x1",
    "write_sparameters_grating_coupler",
    "write_sparameters_grating_coupler_batch",
]

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import ubcpdk

    fiber_angle_deg = +31.0
    c = ubcpdk.components.gc_te1550()
    sim = get_simulation_grating_coupler(
        c,
        is_3d=False,
        fiber_angle_deg=fiber_angle_deg,
        fiber_xoffset=25,
    )
    # f = plot_simulation(sim)

    offsets = np.arange(-10, 11, 2)
    jobs = [
        dict(
            component=c,
            is_3d=False,
            fiber_angle_deg=fiber_angle_deg,
            fiber_xoffset=25 + fiber_xoffset,
        )
        for fiber_xoffset in offsets
    ]
    dfs = write_sparameters_grating_coupler_batch(jobs)

    def log(x):
        return 20 * np.log10(x)

    for offset in offsets:
        df = write_sparameters_grating_coupler(
            c, is_3d=False, fiber_angle_deg=fiber_angle_deg, fiber_xoffset=25 + offset
        )
        plt.plot(df.wavelengths, log(df.s21m), label=str(offset))

    plt.xlabel("wavelength (um")
    plt.ylabel("Transmission (dB)")
    plt.title("transmission vs xoffset")
    plt.legend()
    plt.show()
