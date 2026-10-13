import random

from isaacsim.core.prims import RigidPrim
from isaacsim.core.prims.impl.single_prim_wrapper import _SinglePrimWrapper
from isaacsim.core.prims.impl.xform_prim import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
import numpy as np
from omegaconf import DictConfig
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import torch
import transforms3d as t3d

from env.environment.isaac.physx_device import align_isaacsim_view_to_physx, as_numpy
from env.scene_manager.layout_manager import LayoutManager
from env.scene_manager.objects.action_graph_paths import remap_referenced_action_graph_prim_paths
from env.scene_manager.objects.env_origin import shift_by_origin, store_env_origin


class DynamicObject(_SinglePrimWrapper):
    """USD wrapper that keeps the asset untouched apart from root xform placement."""

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
        if not usd_path:
            raise ValueError("DynamicObject requires a valid usd_path.")

        prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Failed to load USD from {usd_path} to {prim_path}")
        remap_referenced_action_graph_prim_paths(instance_root_prim=prim, source_usd_path=usd_path)

        self.stage = get_current_stage()
        self._prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        try:
            self.model_name, self.model_id = LayoutManager.decode_inst_name(self.instance_name)
        except Exception:
            self.model_name, self.model_id = self.instance_name, None

        self.instance_config = inst_config
        self.visual_config = self.instance_config.get("visual", {})
        self.physics_config = self.instance_config.get("physics", {})
        self.env_origin = store_env_origin(env_origin)

        self.usd_path = usd_path
        self.usd_prim_path = prim_path
        self.default_pos = default_pos
        self.default_ori = default_ori
        self.scale = scale
        self.visible = self.visual_config.get("visible", None)
        self._rigid_body_prim_path = self._find_first_rigid_body_prim_path(prim)
        self._rigid_body_view = None
        self._rigid_body_default_local_positions = None
        self._rigid_body_default_local_orientations = None
        if self._rigid_body_prim_path is not None:
            body_prim = self.stage.GetPrimAtPath(self._rigid_body_prim_path)
            local_transform = UsdGeom.Xformable(body_prim).GetLocalTransformation()
            if isinstance(local_transform, tuple):
                local_transform = local_transform[0]
            transform = Gf.Transform(local_transform)
            translation = transform.GetTranslation()
            quaternion = transform.GetRotation().GetQuat()
            self._rigid_body_default_local_positions = np.asarray([translation], dtype=np.float32)
            self._rigid_body_default_local_orientations = np.asarray(
                [[quaternion.GetReal(), *quaternion.GetImaginary()]], dtype=np.float32
            )

        self._backend = SimulationManager.get_backend()
        self._device = SimulationManager.get_physics_sim_device()
        self._backend_utils = SimulationManager._get_backend_utils()

        translation = self._backend_utils.expand_dims(self._backend_utils.convert(self.default_pos, self._device), 0)
        orientation = self._backend_utils.expand_dims(self._backend_utils.convert(self.default_ori, self._device), 0)
        local_scale = self._backend_utils.expand_dims(self._backend_utils.convert(self.scale, self._device), 0)
        self._xform_prim_view = XFormPrim(
            prim_paths_expr=prim_path,
            name=self.instance_name,
            translations=translation,
            orientations=orientation,
            scales=local_scale,
            reset_xform_properties=True,
        )
        super().__init__(view=self._xform_prim_view)

    def initialize(self):
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        super().initialize(physics_sim_view=self.physics_sim_view)
        align_isaacsim_view_to_physx(self)

    @staticmethod
    def _find_first_rigid_body_prim_path(root_prim) -> str | None:
        for prim in Usd.PrimRange(root_prim):
            if prim != root_prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim.GetPath().pathString
        return None

    def _ensure_rigid_body_view(self) -> bool:
        """Bind the driven body after PhysX has parsed the referenced USD."""
        if self._rigid_body_view is None:
            if self._rigid_body_prim_path is None:
                return False
            self._rigid_body_view = RigidPrim(
                prim_paths_expr=self._rigid_body_prim_path,
                name=f"{self.instance_name}_rigid_body",
                reset_xform_properties=False,
                prepare_contact_sensors=False,
            )
        # The PhysX handle can become available after the view is constructed.
        align_isaacsim_view_to_physx(self._rigid_body_view)
        return True

    def _reset_rigid_body_states(self):
        """Restore the authored child-body pose and clear linear/angular speed."""
        if not self._ensure_rigid_body_view():
            return
        body_device = self._rigid_body_view._device
        self._rigid_body_view.set_local_poses(
            translations=self._backend_utils.convert(self._rigid_body_default_local_positions, body_device),
            orientations=self._backend_utils.convert(self._rigid_body_default_local_orientations, body_device),
        )
        velocities = self._backend_utils.create_zeros_tensor(
            [self._rigid_body_view.count, 6], dtype="float32", device=body_device
        )
        self._rigid_body_view.set_velocities(velocities)

    def get_physics_root_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the asset-root pose from its live driven-body pose.

        Metadata bboxes use the referenced root frame; the root Xform itself
        remains at the spawn pose while PhysX moves the child body.
        """
        if not self._ensure_rigid_body_view():
            raise RuntimeError(f"Dynamic object {self.instance_name!r} has no descendant rigid body.")
        if not self._rigid_body_view.is_physics_handle_valid():
            raise RuntimeError(f"Dynamic object {self.instance_name!r} has no valid PhysX body view.")
        positions, orientations = self._rigid_body_view.get_world_poses()
        positions = as_numpy(positions).reshape(-1, 3)
        orientations = as_numpy(orientations).reshape(-1, 4)
        if positions.shape[0] != 1 or orientations.shape[0] != 1:
            raise RuntimeError(f"Dynamic object {self.instance_name!r} must resolve exactly one driven rigid body.")
        body_rotation_world = t3d.quaternions.quat2mat(orientations[0].astype(float, copy=False))
        body_rotation_local = t3d.quaternions.quat2mat(
            np.asarray(self._rigid_body_default_local_orientations[0], dtype=float)
        )
        root_rotation_world = body_rotation_world @ body_rotation_local.T
        body_position_local = np.asarray(self._rigid_body_default_local_positions[0], dtype=float)
        root_position_world = positions[0] - root_rotation_world @ (
            body_position_local * np.asarray(self.scale, dtype=float)
        )
        return root_position_world, t3d.quaternions.mat2quat(root_rotation_world)

    def _clear_persistent_wrenches(self):
        """Clear Action Graph force/torque values before an unrendered reset."""
        if self._rigid_body_prim_path is None:
            return
        body_prim = self.stage.GetPrimAtPath(self._rigid_body_prim_path)
        if not body_prim or not body_prim.IsValid():
            return
        for name in ("physxForce:force", "physxForce:torque"):
            attribute = body_prim.GetAttribute(name)
            if attribute and attribute.IsValid():
                attribute.Set((0.0, 0.0, 0.0))

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str | None:
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None
        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString
        return None

    def sample_mesh_vertices(self):
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

    def get_bbox(self, is_relative: bool = True):
        points_world, points_local, pos_world, ori_world = self.sample_mesh_vertices()
        if pos_world is None or ori_world is None or points_local.size == 0:
            pos_world, ori_world = self.get_world_pose()
            pos_world = as_numpy(pos_world, dtype=np.float32)
            ori_world = as_numpy(ori_world, dtype=np.float32)
            bbox = np.zeros(6, dtype=np.float32)
        else:
            bbox = np.concatenate([points_local.min(axis=0), points_local.max(axis=0)])

        pos = shift_by_origin(pos_world, self.env_origin, relative=is_relative)
        return pos, ori_world, bbox

    def apply_saved_pose(self):
        self.set_local_pose(translation=self.default_pos, orientation=self.default_ori)
        self.set_local_scale(np.array(self.scale))
        self._clear_persistent_wrenches()
        self._reset_rigid_body_states()

    def relocate_offscreen(self):
        _FAR_CENTER = (100000.0, 100000.0, 100000.0)
        _FAR_JITTER = 1000.0
        far_pos = (
            _FAR_CENTER[0] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[1] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[2] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
        )
        self.set_local_pose(translation=far_pos, orientation=self.default_ori)
        self.set_local_scale(np.array(self.scale))
        self._clear_persistent_wrenches()
        self._reset_rigid_body_states()
