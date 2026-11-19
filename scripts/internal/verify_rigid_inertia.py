"""Check that configured mass changes reach PhysX together with inertia.

The experiment uses two identical one-unit cubes. The heavier cube has 3.5 times
the mass, receives the same off-center force, and should therefore have 3.5 times
the diagonal inertia and 1/3.5 of the angular velocity increment.

Run this file with the RoboDojo Isaac Sim environment, for example:

    PYTHONPATH=. python scripts/internal/verify_rigid_inertia.py
"""

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True, "renderer": "None"})

import numpy as np
import omni.usd
import torch
from isaacsim.core.api import World
from omegaconf import OmegaConf
from pxr import Gf, UsdGeom, UsdPhysics

from env.scene_manager.objects.rigid import RigidObject


MASS_RATIO = 3.5
CUBE_INERTIA_PER_KILOGRAM = 1.0 / 6.0
APPLIED_FORCE = np.asarray([[0.0, 10.0, 0.0]], dtype=np.float32)
FORCE_POSITION = np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32)


def define_authored_cube(stage, prim_path):
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr().Set(1.0)
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(1.0)
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(
            CUBE_INERTIA_PER_KILOGRAM,
            CUBE_INERTIA_PER_KILOGRAM,
            CUBE_INERTIA_PER_KILOGRAM,
        )
    )


def create_rigid_object(stage, prim_path, mass, position):
    define_authored_cube(stage, prim_path)
    config = OmegaConf.create(
        {
            "physics": {"mass": mass},
            "visual": {"visible": False},
        }
    )
    return RigidObject(
        prim_path=prim_path,
        usd_path=None,
        inst_config=config,
        env_origin=torch.zeros(3),
        default_pos=position,
        default_ori=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )


def read_runtime_state(rigid_object):
    mass = float(np.asarray(rigid_object._rigid_prim_view.get_masses()).reshape(-1)[0])
    inertia_matrix = np.asarray(rigid_object._rigid_prim_view.get_inertias()).reshape(-1, 9)[0].reshape(3, 3)
    velocities = np.asarray(rigid_object._rigid_prim_view.get_velocities()).reshape(-1, 6)[0]
    return mass, np.diag(inertia_matrix), velocities[3:]


def main():
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, backend="numpy", device="cpu")

    light = create_rigid_object(
        stage,
        "/World/envs/env_0/rigid/inertia_test/light_0_100",
        mass=1.0,
        position=(-2.0, 0.0, 0.0),
    )
    heavy = create_rigid_object(
        stage,
        "/World/envs/env_0/rigid/inertia_test/heavy_0_100",
        mass=MASS_RATIO,
        position=(2.0, 0.0, 0.0),
    )

    try:
        world.reset()
        simulation_app.update()
        light.initialize()
        heavy.initialize()
        light._rigid_prim_view.disable_gravities()
        heavy._rigid_prim_view.disable_gravities()
        world.step(render=False)
        simulation_app.update()

        light_mass, light_inertia, _ = read_runtime_state(light)
        heavy_mass, heavy_inertia, _ = read_runtime_state(heavy)
        light._rigid_prim_view.set_velocities(np.zeros((1, 6), dtype=np.float32))
        heavy._rigid_prim_view.set_velocities(np.zeros((1, 6), dtype=np.float32))

        for rigid_object in (light, heavy):
            rigid_object._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=APPLIED_FORCE,
                positions=FORCE_POSITION,
                indices=np.asarray([0], dtype=np.int32),
                is_global=False,
            )

        world.step(render=False)
        simulation_app.update()
        _, _, light_angular_velocity = read_runtime_state(light)
        _, _, heavy_angular_velocity = read_runtime_state(heavy)

        inertia_ratio = heavy_inertia / light_inertia
        angular_velocity_ratio = light_angular_velocity[2] / heavy_angular_velocity[2]

        print(f"light_mass={light_mass:.7f}", flush=True)
        print(f"heavy_mass={heavy_mass:.7f}", flush=True)
        print(f"light_inertia={light_inertia.tolist()}", flush=True)
        print(f"heavy_inertia={heavy_inertia.tolist()}", flush=True)
        print(f"inertia_ratio={inertia_ratio.tolist()}", flush=True)
        print(f"light_angular_velocity_z={light_angular_velocity[2]:.7f}", flush=True)
        print(f"heavy_angular_velocity_z={heavy_angular_velocity[2]:.7f}", flush=True)
        print(f"angular_velocity_ratio={angular_velocity_ratio:.7f}", flush=True)

        np.testing.assert_allclose(heavy_mass / light_mass, MASS_RATIO, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(inertia_ratio, np.full(3, MASS_RATIO), rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(angular_velocity_ratio, MASS_RATIO, rtol=5e-2, atol=5e-2)
        print("PASS: mass, inertia, and angular response scale consistently", flush=True)
    finally:
        world.clear_instance()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
