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
from env.scene_manager.objects.physics_utils import (
    disable_nested_rigid_bodies,
    enable_collision_tree,
    enable_contact_report,
    get_or_apply_api,
    set_kinematic_body,
)

MIN_RIGID_MASS = 0.05
MAX_RIGID_MASS = 0.5


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
        # Read any authored USD MassAPI properties before overriding them so
        # fallback mass resolution and inertia scaling use the asset's true values.
        self._mass_api_prim = self._find_mass_api_prim()
        self._asset_mass = self._read_asset_mass()
        self._asset_diagonal_inertia = self._read_asset_diagonal_inertia()
        self._asset_center_of_mass = self._read_mass_attribute("GetCenterOfMassAttr")
        self._asset_principal_axes = self._read_mass_attribute("GetPrincipalAxesAttr")

        configured_mass = self.physics_config.get("mass", None)
        if configured_mass is not None:
            self.mass = self._clamp_mass(configured_mass)
        elif self._asset_mass is not None and self._asset_mass > 0.0:
            self.mass = self._clamp_mass(self._asset_mass)
        else:
            self.mass = MAX_RIGID_MASS
        self.visible = self.visual_config.get("visible", True)

        self.physics_material_path = find_unique_string_name(
            prim_path + "/physics_material",
            lambda x: not is_prim_path_valid(x),
        )
        self.physics_material = PhysicsMaterial(
            prim_path=self.physics_material_path,
            static_friction=self.physics_config.get(
                "static_friction",
                0.5,
            ),
            dynamic_friction=self.physics_config.get(
                "dynamic_friction",
                0.5,
            ),
            restitution=self.physics_config.get("restitution", 0.0),
            config=self.physics_config,
        )

        mass_api = get_or_apply_api(prim, UsdPhysics.MassAPI)
        mass_api.CreateMassAttr().Set(self.mass)
        if self._asset_mass is not None and self._asset_mass > 0.0 and self._asset_diagonal_inertia is not None:
            inertia_scale = self.mass / self._asset_mass
            scaled_diagonal_inertia = np.asarray(self._asset_diagonal_inertia, dtype=np.float32) * inertia_scale
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*scaled_diagonal_inertia.tolist()))
            if self._asset_center_of_mass is not None:
                mass_api.CreateCenterOfMassAttr().Set(self._asset_center_of_mass)
            if self._asset_principal_axes is not None:
                mass_api.CreatePrincipalAxesAttr().Set(self._asset_principal_axes)
        self._mass_api_prim = prim

        # Write the USD binding before either base class can flush a newly created PhysX handle.
        # If an actor already exists before this wrapper is constructed, it must be created pre-reset.
        material_binding = get_or_apply_api(prim, UsdShade.MaterialBindingAPI)
        material_binding.Bind(
            self.physics_material.material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )

        collision = bool(self.physics_config.get("collision", False))
        SingleGeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            scale=self.scale,
            visible=self.visible,
            collision=collision,
            track_contact_forces=False,
        )
        if collision:
            enable_collision_tree(prim)

        disable_nested_rigid_bodies(prim)
        SingleRigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            translation=self.default_pos,
            orientation=self.default_ori,
            scale=self.scale,
        )
        root_prim = self.stage.GetPrimAtPath(self._prim_path)
        if self.physics_config.get("kinematic", False):
            set_kinematic_body(root_prim)
        if self.physics_config.get("contact_report", False):
            # Dynamic bodies report contacts on the rigid-body prim.
            enable_contact_report(root_prim, threshold=0.0, include_colliders=False)
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
            return None

        mass_api = UsdPhysics.MassAPI.Get(self.stage, self._mass_api_prim.GetPath())
        mass_value = mass_api.GetMassAttr().Get()
        if mass_value is None or float(mass_value) <= 0.0:
            return None
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
    def _to_numpy(value) -> np.ndarray:
        """Convert a backend tensor/array to a CPU NumPy array."""
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _match_backend_tensor(values: np.ndarray, template):
        """Convert a NumPy array to match the PhysX view's expected tensor/array type."""
        if isinstance(template, torch.Tensor):
            return torch.as_tensor(values, dtype=template.dtype, device=template.device).reshape(template.shape)
        template_arr = np.asarray(template)
        return np.asarray(values, dtype=template_arr.dtype).reshape(template_arr.shape)

    @staticmethod
    def _usd_inertia_attributes_to_matrix(diagonal_inertia, principal_axes=None) -> np.ndarray:
        """Reconstruct a 3x3 body-frame inertia tensor from USD diagonalInertia and principalAxes."""
        diag = np.asarray(diagonal_inertia, dtype=np.float64).reshape(3)
        if principal_axes is None:
            return np.diag(diag)
        if hasattr(principal_axes, "GetReal"):
            w = float(principal_axes.GetReal())
            imag = principal_axes.GetImaginary()
            x, y, z = float(imag[0]), float(imag[1]), float(imag[2])
        else:
            quat = np.asarray(principal_axes, dtype=np.float64).reshape(-1)
            if quat.size < 4:
                return np.diag(diag)
            w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-12:
            return np.diag(diag)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        rot = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return rot @ np.diag(diag) @ rot.T

    @staticmethod
    def _inertia_matrix_to_usd_attributes(inertia_matrix):
        """Convert a body-frame inertia matrix to USD principal inertia attributes."""
        inertia_matrix = RigidObject._to_numpy(inertia_matrix)
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
        """Keep runtime PhysX and USD mass/inertia consistent with ``self.mass``."""
        if self._mass_inertia_synced:
            return

        view = getattr(self, "_rigid_prim_view", None)
        if view is None or not view.is_physics_handle_valid():
            return

        raw_masses = view.get_masses()
        masses_np = self._to_numpy(raw_masses).astype(np.float32, copy=True)
        current_mass = float(masses_np.reshape(-1)[0])
        masses_np.reshape(-1)[0] = self.mass
        view.set_masses(self._match_backend_tensor(masses_np, raw_masses))

        raw_inertias = view.get_inertias()
        inertias_np = self._to_numpy(raw_inertias).astype(np.float32, copy=True).reshape(view.count, 9)
        has_authored_inertia = (
            self._asset_diagonal_inertia is not None and self._asset_mass is not None and self._asset_mass > 0.0
        )
        if has_authored_inertia:
            ratio = self.mass / self._asset_mass
            target_diag = np.asarray(self._asset_diagonal_inertia, dtype=np.float64) * ratio
            target_mat = self._usd_inertia_attributes_to_matrix(target_diag, self._asset_principal_axes)
            inertias_np[0] = target_mat.reshape(9).astype(np.float32)
        elif current_mass > 0.0 and not np.isclose(current_mass, self.mass, rtol=1e-4, atol=1e-6):
            inertias_np[0] *= float(self.mass / current_mass)
        view.set_inertias(self._match_backend_tensor(inertias_np, raw_inertias))

        mass_prim = self._mass_api_prim
        if mass_prim is None:
            mass_prim = self.stage.GetPrimAtPath(self._prim_path)
            self._mass_api_prim = mass_prim
        if mass_prim is not None and mass_prim.IsValid():
            mass_api = get_or_apply_api(mass_prim, UsdPhysics.MassAPI)
            mass_api.CreateMassAttr().Set(self.mass)
            if has_authored_inertia:
                mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(value) for value in target_diag]))
                if self._asset_center_of_mass is not None:
                    mass_api.CreateCenterOfMassAttr().Set(self._asset_center_of_mass)
                if self._asset_principal_axes is not None:
                    mass_api.CreatePrincipalAxesAttr().Set(self._asset_principal_axes)
            else:
                target_diagonal_inertia, target_principal_axes = self._inertia_matrix_to_usd_attributes(inertias_np[0])
                mass_api.CreateDiagonalInertiaAttr().Set(target_diagonal_inertia)
                mass_api.CreatePrincipalAxesAttr().Set(target_principal_axes)
                com_positions, _ = view.get_coms()
                center_of_mass = self._to_numpy(com_positions[0]).astype(np.float32).reshape(-1)[:3]
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
        self._sync_mass_properties()

        if self._default_linear_velocity is not None or self._default_angular_velocity is not None:
            self.set_default_state(
                linear_velocity=self._default_linear_velocity,
                angular_velocity=self._default_angular_velocity,
            )

    def set_mass(self, mass: float) -> None:
        """Set mass and scale inertia consistently for this rigid body."""
        mass = self._clamp_mass(mass)

        rigid_view = getattr(self, "_rigid_prim_view", None)
        if rigid_view is None or not rigid_view.is_physics_handle_valid():
            prim = self.stage.GetPrimAtPath(self._prim_path)
            mass_api = get_or_apply_api(prim, UsdPhysics.MassAPI)
            previous_mass = mass_api.GetMassAttr().Get()
            previous_inertia = mass_api.GetDiagonalInertiaAttr().Get()
            SingleRigidPrim.set_mass(self, mass)
            mass_api.CreateMassAttr().Set(mass)
            if (
                previous_mass
                and float(previous_mass) > 0.0
                and previous_inertia is not None
                and any(float(value) > 0.0 for value in previous_inertia)
            ):
                scaled_inertia = np.asarray(previous_inertia, dtype=np.float32) * (mass / float(previous_mass))
                mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*scaled_inertia.tolist()))
            self.mass = mass
            return

        raw_masses = rigid_view.get_masses()
        raw_inertias = rigid_view.get_inertias()
        masses_np = self._to_numpy(raw_masses).astype(np.float32, copy=True)
        previous_mass = float(masses_np.reshape(-1)[0])
        masses_np.reshape(-1)[0] = mass
        rigid_view.set_masses(self._match_backend_tensor(masses_np, raw_masses))

        inertias_np = self._to_numpy(raw_inertias).astype(np.float32, copy=True).reshape(rigid_view.count, 9)
        if previous_mass > 0.0:
            inertias_np[0] *= float(mass / previous_mass)
            rigid_view.set_inertias(self._match_backend_tensor(inertias_np, raw_inertias))

        prim = self.stage.GetPrimAtPath(self._prim_path)
        mass_api = get_or_apply_api(prim, UsdPhysics.MassAPI)
        mass_api.CreateMassAttr().Set(mass)
        if previous_mass > 0.0:
            target_diagonal_inertia, target_principal_axes = self._inertia_matrix_to_usd_attributes(inertias_np[0])
            mass_api.CreateDiagonalInertiaAttr().Set(target_diagonal_inertia)
            mass_api.CreatePrincipalAxesAttr().Set(target_principal_axes)
        self.mass = mass
        self._mass_inertia_synced = True

    @staticmethod
    def _clamp_mass(mass: float) -> float:
        """Keep rigid-object mass within the benchmark's supported range."""
        return min(max(float(mass), MIN_RIGID_MASS), MAX_RIGID_MASS)

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
