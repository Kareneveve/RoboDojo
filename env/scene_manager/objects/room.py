from __future__ import annotations

from typing import Any, Dict

from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


class Room:
    """Load a composed room reference while preserving its authored colliders."""

    def __init__(
        self,
        prim_path: str | None = None,
        usd_path: str | None = None,
        instance_config: Dict[str, Any] | None = None,
    ):
        """
        Initialize Room object wrapping the room geometry prim.

        Args:
            prim_path: Prim path in Isaac Sim (optional)
            usd_path: USD asset file path (optional, if None will wrap existing prim)
            instance_config: Instance configuration dictionary (optional)
        """
        if prim_path is not None:
            if usd_path is not None:
                room_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                room_prim = get_current_stage().GetPrimAtPath(prim_path)
            if not room_prim or not room_prim.IsValid():
                raise RuntimeError(f"Failed to load room from {usd_path} at {prim_path}")

            # Room roots are Xforms, not geometry. Keep collision schemas on the
            # referenced meshes instead of applying them to the reference root.
            self.prim = room_prim
            self.prim_path = prim_path
            self.name = prim_path.split("/")[-1]

            if instance_config:
                path_parts = prim_path.split("/")
                self.instance_name = path_parts[-1]
                self.category_name = self.instance_name.rsplit("_", 1)[0]
                self.default_pos = instance_config.get("default_pos", [0.0, 0.0, 0.0])
                self.default_ori = instance_config.get("default_ori", [1.0, 0.0, 0.0, 0.0])
                self.scale = instance_config.get("scale", [1.0, 1.0, 1.0])

                scale_matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float, self.scale)))
                rotation_matrix = Gf.Matrix4d().SetRotate(
                    Gf.Quatd(float(self.default_ori[0]), Gf.Vec3d(*map(float, self.default_ori[1:4])))
                )
                transform = scale_matrix * rotation_matrix
                transform.SetTranslateOnly(Gf.Vec3d(*map(float, self.default_pos)))
                UsdGeom.Xformable(room_prim).MakeMatrixXform().Set(transform)

                if not instance_config.get("collision", True):
                    for prim in Usd.PrimRange(room_prim):
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                self._apply_object_translations(instance_config.get("object_translations", {}))

    def _apply_object_translations(self, entries):
        """Apply saved room-frame furniture offsets to this room instance."""
        if not entries:
            return
        stage = self.prim.GetStage()
        time = Usd.TimeCode.Default()
        room_world = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(time)
        for entry in entries.values():
            path = entry["prim_path"]
            if not path or path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")):
                raise ValueError(f"Room translation requires a relative child prim path: {path}")
            prim = stage.GetPrimAtPath(f"{self.prim_path}/{path}")
            if not prim.IsValid() or not UsdGeom.Xformable(prim):
                raise ValueError(f"Room translation prim does not exist or is not transformable: {path}")
            offset = np.asarray(entry["translation"], dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError(f"Invalid room object translation: {path}")
            xform = UsdGeom.Xformable(prim)
            parent_world = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(time)
            parent_to_room = parent_world * room_world.GetInverse()
            local_offset = parent_to_room.GetInverse().TransformDir(Gf.Vec3d(*map(float, offset)))
            matrix = Gf.Matrix4d(xform.GetLocalTransformation())
            matrix.SetTranslateOnly(matrix.ExtractTranslation() + local_offset)
            xform.MakeMatrixXform().Set(matrix)
