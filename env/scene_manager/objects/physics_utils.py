"""USD physics authoring helpers shared by scene-object wrappers.

Object wrappers (``GeometryObject``, ``RigidObject``) translate the optional
``physics`` keys of an instance config into USD/PhysX schema attributes
through these helpers, so tasks only declare *what* they need (for example via
``SceneManager.set_physics_overrides``) instead of editing prims directly.

Supported ``physics`` keys:

``collision``       Enable the root collider and every authored collider below it.
``kinematic``       Give the prim an enabled, kinematic rigid body.  The object
                    stays fixed under contacts but participates in PhysX
                    contact queries (needed for GPU ``RigidContactView``).
``contact_report``  Apply ``PhysxContactReportAPI`` with a zero threshold.
"""

from pxr import PhysxSchema, Usd, UsdPhysics


def get_or_apply_api(prim, api_cls):
    """Return an existing USD/PhysX schema API on ``prim``, applying it if absent."""
    return api_cls(prim) if prim.HasAPI(api_cls) else api_cls.Apply(prim)


def iter_collider_prims(root_prim):
    """Yield every prim under ``root_prim`` (inclusive) that has a CollisionAPI."""
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            yield prim


def enable_collision_tree(root_prim):
    """Enable collision on ``root_prim`` and every authored collider below it."""
    get_or_apply_api(root_prim, UsdPhysics.CollisionAPI).CreateCollisionEnabledAttr().Set(True)
    for prim in iter_collider_prims(root_prim):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)


def set_kinematic_body(prim):
    """Author an enabled kinematic rigid body on ``prim``."""
    rigid_body_api = get_or_apply_api(prim, UsdPhysics.RigidBodyAPI)
    rigid_body_api.CreateRigidBodyEnabledAttr().Set(True)
    rigid_body_api.CreateKinematicEnabledAttr().Set(True)


def enable_contact_report(root_prim, threshold=0.0, include_colliders=False):
    """Apply ``PhysxContactReportAPI`` to ``root_prim``.

    Args:
        root_prim: Object root prim.
        threshold: Contact report force threshold.
        include_colliders: Also apply the API to collider prims below the root.
            Static/kinematic geometry reports per collider shape, so geometry
            objects enable this; dynamic rigid bodies report on the body prim.
    """
    targets = [root_prim]
    if include_colliders:
        targets.extend(prim for prim in iter_collider_prims(root_prim) if prim != root_prim)
    for prim in targets:
        get_or_apply_api(prim, PhysxSchema.PhysxContactReportAPI).CreateThresholdAttr().Set(float(threshold))
