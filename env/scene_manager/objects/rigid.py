import random

from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
import numpy as np
from omegaconf import DictConfig
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial import ConvexHull, QhullError
import torch

from env.scene_manager.layout_manager import LayoutManager
from env.scene_manager.objects.physics_material import PhysicsMaterial


class RigidObject(SingleRigidPrim, SingleGeometryPrim):
    """Rigid body object with physical properties and collision detection.
    Combines geometry (visual/collision) and rigid body dynamics.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        inst_config: DictConfig,
        env_origin: torch.Tensor,
        default_pos: tuple,
        default_ori: tuple,
        scale: tuple,
    ):
        if usd_path:
            prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        else:
            prim = get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            error_message = (
                f"Failed to load USD from {usd_path} to {prim_path}"
                if usd_path
                else f"Failed to find an existing prim at path {prim_path}"
            )
            raise RuntimeError(error_message)

        """Initialize a rigid body object with geometry and physics properties."""
        self.stage = prim.GetStage()
        self._prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.model_name, self.model_id = LayoutManager.decode_inst_name(self.instance_name)

        self.instance_config = inst_config
        self.visual_config = self.instance_config.get("visual", {})
        self.physics_config = self.instance_config.get("physics", {})

        self.env_origin = env_origin
        if isinstance(self.env_origin, torch.Tensor):
            self.env_origin = self.env_origin.cpu().numpy().tolist()
        self.usd_path = usd_path
        self.usd_prim_path = prim_path
        self.default_pos = default_pos
        self.default_ori = default_ori
        self.scale = scale
        # Resolve the configured mass after reading the authored USD mass.
        # Keep a positive fallback during construction because the USD mass
        # API is inspected below, before the PhysX view is initialized.
        configured_mass = self.physics_config.get("mass", None)
        self.mass = float(configured_mass) if configured_mass is not None else 0.5
        self.visible = self.visual_config.get("visible", True)

        self.physics_material_path = find_unique_string_name(
            prim_path + "/physics_material",
            lambda x: not is_prim_path_valid(x),
        )
        self.physics_material = PhysicsMaterial(
            prim_path=self.physics_material_path,
            static_friction=self.physics_config.get(
                "static_friction",
                self.physics_config.get("friction", 0.6),
            ),
            dynamic_friction=self.physics_config.get(
                "dynamic_friction",
                self.physics_config.get("friction", 1.5),
            ),
            restitution=self.physics_config.get("restitution", 0.0),
            config=self.physics_config,
        )

        mass_api = (
            UsdPhysics.MassAPI(prim)
            if prim.HasAPI(UsdPhysics.MassAPI)
            else UsdPhysics.MassAPI.Apply(prim)
        )
        authored_mass = mass_api.GetMassAttr().Get()
        authored_diagonal_inertia = mass_api.GetDiagonalInertiaAttr().Get()
        self._authored_mass_before_override = float(authored_mass) if authored_mass is not None else None
        self._authored_diagonal_inertia_before_override = (
            np.asarray(authored_diagonal_inertia, dtype=np.float32)
            if authored_diagonal_inertia is not None
            else None
        )
        mass_api.CreateMassAttr().Set(self.mass)
        if (
            self._authored_mass_before_override
            and self._authored_diagonal_inertia_before_override is not None
        ):
            inertia_scale = self.mass / self._authored_mass_before_override
            scaled_diagonal_inertia = self._authored_diagonal_inertia_before_override * inertia_scale
            mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*scaled_diagonal_inertia.tolist()))

        # Write the USD binding before either base class can flush a newly created PhysX handle.
        # If an actor already exists before this wrapper is constructed, it must be created pre-reset.
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            material_binding = UsdShade.MaterialBindingAPI(prim)
        else:
            material_binding = UsdShade.MaterialBindingAPI.Apply(prim)
        material_binding.Bind(
            self.physics_material.material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )

        SingleGeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            scale=self.scale,
            visible=self.visible,
            collision=False,
            track_contact_forces=False,
        )

        SingleRigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            translation=self.default_pos,
            orientation=self.default_ori,
            scale=self.scale,
        )
        self._mass_api_prim = self._find_mass_api_prim()
        self._asset_mass = self._read_asset_mass()
        if configured_mass is None and self._asset_mass is not None and self._asset_mass > 0:
            self.mass = float(self._asset_mass)
        self._asset_diagonal_inertia = self._read_asset_diagonal_inertia()
        self._asset_center_of_mass = self._read_mass_attribute("GetCenterOfMassAttr")
        self._asset_principal_axes = self._read_mass_attribute("GetPrincipalAxesAttr")
        self._mass_inertia_synced = False
        self._default_linear_velocity = [0.0, 0.0, 0.0]
        self._default_angular_velocity = [0.0, 0.0, 0.0]

        self._setup_physics()

    def _get_object_transform(self, device=None):
        """
        Get Pose and Orientation Matrix

        Returns:
            Tuple of (obj_pos, obj_quat, device)
        """
        obj_translation, obj_orientation = self.get_local_pose()

        if device is None:
            if isinstance(obj_translation, torch.Tensor):
                device = obj_translation.device
            elif isinstance(obj_orientation, torch.Tensor):
                device = obj_orientation.device
            else:
                device = "cpu"

        if isinstance(obj_translation, torch.Tensor):
            obj_pos = obj_translation.to(dtype=torch.float32, device=device).squeeze()[:3]
        else:
            obj_pos = torch.tensor(obj_translation, dtype=torch.float32, device=device)[:3]

        if isinstance(obj_orientation, torch.Tensor):
            obj_quat = obj_orientation.to(dtype=torch.float32, device=device).squeeze()[:4]
        else:
            obj_quat = torch.tensor(obj_orientation, dtype=torch.float32, device=device)[:4]

        return obj_pos, obj_quat, device

    @property
    def local_collision_vertices(self):
        """Return collision mesh vertices in the scaled object frame."""
        root = self.stage.GetPrimAtPath(self._prim_path)
        xforms = UsdGeom.XformCache()
        mesh_vertices = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if not points:
                continue
            transform, _ = xforms.ComputeRelativeTransform(prim, root)
            matrix = np.asarray(transform)
            mesh_vertices.append(np.asarray(points) @ matrix[:3, :3] + matrix[3, :3])

        if not mesh_vertices:
            return np.empty((0, 3))

        vertices = np.concatenate(mesh_vertices) * np.asarray(self.scale, dtype=float)
        # Extrema along any axis are attained on the hull even as the object rotates.
        try:
            return vertices[ConvexHull(vertices).vertices]
        except QhullError:
            return vertices

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None
        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString
        return None

    def _find_mass_api_prim(self):
        """Find the asset prim carrying the authored mass properties."""
        root_prim = self.stage.GetPrimAtPath(self._prim_path)
        if not root_prim or not root_prim.IsValid():
            return None

        if root_prim.HasAPI(UsdPhysics.MassAPI):
            return root_prim

        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.MassAPI):
                return prim
        return None

    def _read_asset_mass(self):
        """Read the authored asset mass before the configured mass is applied."""
        if self._mass_api_prim is None:
            return self.mass

        mass_api = UsdPhysics.MassAPI.Get(self.stage, self._mass_api_prim.GetPath())
        mass_value = mass_api.GetMassAttr().Get()
        if mass_value is None or float(mass_value) <= 0:
            return self.mass
        return float(mass_value)

    def _read_asset_diagonal_inertia(self):
        """Read the asset's authored diagonal inertia before the override."""
        if self._mass_api_prim is None:
            return None

        mass_api = UsdPhysics.MassAPI.Get(self.stage, self._mass_api_prim.GetPath())
        diagonal_inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if diagonal_inertia is None or not any(float(value) > 0.0 for value in diagonal_inertia):
            return None
        return diagonal_inertia

    def _read_mass_attribute(self, getter_name):
        """Read an authored MassAPI attribute without changing its fallback behavior."""
        if self._mass_api_prim is None:
            return None

        mass_api = UsdPhysics.MassAPI.Get(self.stage, self._mass_api_prim.GetPath())
        return getattr(mass_api, getter_name)().Get()

    @staticmethod
    def _inertia_matrix_to_usd_attributes(inertia_matrix):
        """Convert a body-frame inertia matrix to USD principal inertia attributes."""
        inertia_matrix = np.asarray(inertia_matrix, dtype=np.float64).reshape(3, 3)
        inertia_matrix = 0.5 * (inertia_matrix + inertia_matrix.T)
        diagonal_inertia, principal_axes = np.linalg.eigh(inertia_matrix)
        if np.linalg.det(principal_axes) < 0:
            principal_axes[:, 0] *= -1.0

        trace = np.trace(principal_axes)
        if trace > 0.0:
            w = 0.5 * np.sqrt(trace + 1.0)
            scale = 4.0 * w
            x = (principal_axes[2, 1] - principal_axes[1, 2]) / scale
            y = (principal_axes[0, 2] - principal_axes[2, 0]) / scale
            z = (principal_axes[1, 0] - principal_axes[0, 1]) / scale
        elif principal_axes[0, 0] > principal_axes[1, 1] and principal_axes[0, 0] > principal_axes[2, 2]:
            x = 0.5 * np.sqrt(max(0.0, 1.0 + principal_axes[0, 0] - principal_axes[1, 1] - principal_axes[2, 2]))
            scale = 4.0 * x
            w = (principal_axes[2, 1] - principal_axes[1, 2]) / scale
            y = (principal_axes[0, 1] + principal_axes[1, 0]) / scale
            z = (principal_axes[0, 2] + principal_axes[2, 0]) / scale
        elif principal_axes[1, 1] > principal_axes[2, 2]:
            y = 0.5 * np.sqrt(max(0.0, 1.0 + principal_axes[1, 1] - principal_axes[0, 0] - principal_axes[2, 2]))
            scale = 4.0 * y
            w = (principal_axes[0, 2] - principal_axes[2, 0]) / scale
            x = (principal_axes[0, 1] + principal_axes[1, 0]) / scale
            z = (principal_axes[1, 2] + principal_axes[2, 1]) / scale
        else:
            z = 0.5 * np.sqrt(max(0.0, 1.0 + principal_axes[2, 2] - principal_axes[0, 0] - principal_axes[1, 1]))
            scale = 4.0 * z
            w = (principal_axes[1, 0] - principal_axes[0, 1]) / scale
            x = (principal_axes[0, 2] + principal_axes[2, 0]) / scale
            y = (principal_axes[1, 2] + principal_axes[2, 1]) / scale

        return Gf.Vec3f(*[float(value) for value in diagonal_inertia]), Gf.Quatf(
            float(w), Gf.Vec3f(float(x), float(y), float(z))
        )

    def _sync_mass_properties(self):
        """Keep authored inertia consistent with the configured mass override."""
        if self._mass_inertia_synced:
            return

        view = self._rigid_prim_view
        masses = view.get_masses()
        masses[0] = self.mass
        view.set_masses(masses)

        inertias = view.get_inertias()
        if self._asset_diagonal_inertia is not None:
            ratio = self.mass / self._asset_mass if self._asset_mass > 0 else 1.0
            target_inertias = inertias * ratio
        else:
            target_inertias = inertias
        view.set_inertias(target_inertias)

        if self._mass_api_prim is None:
            self._mass_api_prim = self._find_mass_api_prim()
        if self._mass_api_prim is not None:
            mass_api = UsdPhysics.MassAPI.Get(self.stage, self._mass_api_prim.GetPath())
            mass_api.CreateMassAttr().Set(self.mass)
            if self._asset_diagonal_inertia is not None:
                target_diagonal_inertia = Gf.Vec3f(
                    self._asset_diagonal_inertia[0] * ratio,
                    self._asset_diagonal_inertia[1] * ratio,
                    self._asset_diagonal_inertia[2] * ratio,
                )
                mass_api.CreateDiagonalInertiaAttr().Set(target_diagonal_inertia)
                if self._asset_center_of_mass is not None:
                    mass_api.CreateCenterOfMassAttr().Set(self._asset_center_of_mass)
                if self._asset_principal_axes is not None:
                    mass_api.CreatePrincipalAxesAttr().Set(self._asset_principal_axes)
            else:
                target_diagonal_inertia, target_principal_axes = self._inertia_matrix_to_usd_attributes(
                    target_inertias[0]
                )
                mass_api.CreateDiagonalInertiaAttr().Set(target_diagonal_inertia)
                mass_api.CreatePrincipalAxesAttr().Set(target_principal_axes)
                com_positions, _ = view.get_coms()
                center_of_mass = np.asarray(com_positions[0], dtype=np.float32).reshape(-1)[:3]
                mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(value) for value in center_of_mass]))

        self._mass_inertia_synced = True

    def sample_mesh_vertices(self):
        """
        Get current mesh vertex positions for this rigid object.

        Returns (points_world, points_local, pos_world, ori_world).
        """
        mesh_path = self._find_first_mesh_in_hierarchy(self._prim_path)
        if mesh_path is None:
            return np.array([]), np.array([]), None, None

        mesh_prim = UsdGeom.Mesh.Get(self.stage, mesh_path)
        points_local = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        world_tf = omni.usd.get_world_transform_matrix(mesh_prim.GetPrim())

        points_world = np.array(
            [list(world_tf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))) for p in points_local],
            dtype=np.float32,
        )

        rot_quat = world_tf.ExtractRotationQuat()
        pos_world = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
        ori_world = np.array([rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32)

        return points_world, points_local, pos_world, ori_world

    def get_bbox(self, is_relative=True):
        points_world, points_local, pos_world, ori_world = self.sample_mesh_vertices()

        if is_relative:
            pos = np.subtract(pos_world, self.env_origin)
        else:
            pos = pos_world
        ori = ori_world
        bbox = np.concatenate([points_local.min(axis=0), points_local.max(axis=0)])
        return pos, ori, bbox

    def hide_prim(self, prim_path: str):
        """
        Hide a prim by setting its visibility to invisible.
        This will make the prim and all its children invisible and ignored by physics.

        Args:
            prim_path: The prim path to hide
        """
        try:
            path = Sdf.Path(prim_path)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                print(f"Warning: Invalid prim path {prim_path}")
                return

            # Set visibility to invisible
            visibility_attribute = prim.GetAttribute("visibility")
            if visibility_attribute is None:
                # Create the visibility attribute if it doesn't exist
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
            else:
                visibility_attribute.Set("invisible")

        except Exception as e:
            print(f"Warning: Failed to hide prim {prim_path}: {e}")

    def _setup_physics(self):
        """Configure physics properties (rigid type, mass) from instance config."""
        self._apply_mass_and_inertia()

        if self._default_linear_velocity is not None or self._default_angular_velocity is not None:
            self.set_default_state(
                linear_velocity=self._default_linear_velocity,
                angular_velocity=self._default_angular_velocity,
            )

    def set_mass(self, mass: float) -> None:
        """Set mass and scale inertia consistently for this rigid body."""
        mass = float(mass)
        if mass <= 0.0:
            mass = 0.05

        rigid_view = getattr(self, "_rigid_prim_view", None)
        if rigid_view is None or not rigid_view.is_physics_handle_valid():
            prim = self.stage.GetPrimAtPath(self._prim_path)
            mass_api = (
                UsdPhysics.MassAPI(prim)
                if prim.HasAPI(UsdPhysics.MassAPI)
                else UsdPhysics.MassAPI.Apply(prim)
            )
            previous_mass = mass_api.GetMassAttr().Get()
            previous_inertia = mass_api.GetDiagonalInertiaAttr().Get()
            SingleRigidPrim.set_mass(self, mass)
            if previous_mass and previous_mass > 0.0 and previous_inertia is not None:
                scaled_inertia = np.asarray(previous_inertia, dtype=np.float32) * (mass / previous_mass)
                mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*scaled_inertia.tolist()))
            self.mass = mass
            return

        current_masses = self._backend_utils.to_numpy(rigid_view.get_masses())
        current_inertias = self._backend_utils.to_numpy(rigid_view.get_inertias())
        previous_mass = float(np.asarray(current_masses).reshape(-1)[0])
        inertia_values = np.asarray(current_inertias, dtype=np.float32).reshape(rigid_view.count, 9)
        SingleRigidPrim.set_mass(self, mass)
        if previous_mass > 0.0:
            inertia_values *= mass / previous_mass
            rigid_view.set_inertias(inertia_values)

        prim = self.stage.GetPrimAtPath(self._prim_path)
        mass_api = (
            UsdPhysics.MassAPI(prim)
            if prim.HasAPI(UsdPhysics.MassAPI)
            else UsdPhysics.MassAPI.Apply(prim)
        )
        mass_api.CreateMassAttr().Set(mass)
        if previous_mass > 0.0:
            diagonal_inertia = np.diag(inertia_values[0].reshape(3, 3))
            mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal_inertia.tolist()))
        self.mass = mass

    def _apply_mass_and_inertia(self):
        """Apply configured mass and keep the runtime inertia consistent with it."""
        rigid_view = self._rigid_prim_view
        if not rigid_view.is_physics_handle_valid():
            SingleRigidPrim.set_mass(self, self.mass)
            return

        current_masses = self._backend_utils.to_numpy(rigid_view.get_masses())
        current_inertias = self._backend_utils.to_numpy(rigid_view.get_inertias())
        current_mass = float(np.asarray(current_masses).reshape(-1)[0])
        inertia_values = np.asarray(current_inertias, dtype=np.float32).reshape(rigid_view.count, 9)

        reference_mass = current_mass
        if (
            self._authored_mass_before_override is not None
            and self._authored_diagonal_inertia_before_override is not None
        ):
            reference_mass = self._authored_mass_before_override

        inertia_scale = self.mass / reference_mass if reference_mass > 0.0 else 1.0
        if (
            self._authored_diagonal_inertia_before_override is not None
            and self._authored_mass_before_override
        ):
            expected_diagonal_inertia = self._authored_diagonal_inertia_before_override * inertia_scale
            current_diagonal_inertia = np.diag(inertia_values[0].reshape(3, 3))
            if np.allclose(current_diagonal_inertia, expected_diagonal_inertia, rtol=1e-4, atol=1e-6):
                inertia_scale = 1.0

        SingleRigidPrim.set_mass(self, self.mass)
        if reference_mass > 0.0:
            rigid_view.set_inertias(inertia_values * inertia_scale)

            if (
                self._authored_diagonal_inertia_before_override is not None
                and self._authored_mass_before_override
            ):
                mass_api = UsdPhysics.MassAPI(self.stage.GetPrimAtPath(self._prim_path))
                scaled_diagonal_inertia = self._authored_diagonal_inertia_before_override * (
                    self.mass / self._authored_mass_before_override
                )
                mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*scaled_diagonal_inertia.tolist()))

    def apply_saved_pose(self):
        """Apply the saved eval-layout pose and clear velocities."""
        self.set_local_pose(translation=self.default_pos, orientation=self.default_ori)
        self.set_local_scale(np.array(self.scale))
        self._apply_default_velocities()

    def relocate_offscreen(self):
        """Move this object far away so stale prims do not affect the next layout."""
        _FAR_CENTER = (100000.0, 100000.0, 100000)
        _FAR_JITTER = 1000.0
        far_pos = (
            _FAR_CENTER[0] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[1] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[2] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
        )
        self.set_local_pose(translation=far_pos, orientation=self.default_ori)
        self.set_local_scale(np.array(self.scale))
        self._apply_default_velocities()

    def initialize(self):
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        super().initialize(physics_sim_view=self.physics_sim_view)
        self._sync_mass_properties()

    def _apply_default_velocities(self):
        """Re-apply default linear/angular velocity if configured."""
        if self._default_linear_velocity is not None:
            self.set_linear_velocity(torch.tensor(self._default_linear_velocity))
        if self._default_angular_velocity is not None:
            self.set_angular_velocity(torch.tensor(self._default_angular_velocity))

    def destroy(self):
        self._rigid_prim_view.disable_gravities()
        self._geometry_prim_view.disable_collision()
        self._rigid_prim_view.set_visibilities([False])
        self._geometry_prim_view.set_visibilities([False])
        self.hide_prim(self._prim_path)
