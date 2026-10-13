"""Remap referenced ScriptNode inputs to the spawned Dynamic asset instance."""

from pxr import Sdf, Usd


def _is_action_graph_descendant(prim, instance_root) -> bool:
    ancestor = prim.GetParent()
    while ancestor and ancestor.IsValid() and ancestor != instance_root:
        if ancestor.GetTypeName() == "OmniGraph":
            return True
        ancestor = ancestor.GetParent()
    return False


def remap_referenced_action_graph_prim_paths(*, instance_root_prim, source_usd_path: str) -> None:
    """Override only unconnected string/token prim paths below the asset root.

    USD relationships and connections already remap through references. Plain
    ScriptNode input values do not, so these overrides live in the session layer
    of the instance stage and never modify the source asset.
    """
    if not instance_root_prim or not instance_root_prim.IsValid():
        raise ValueError("instance_root_prim must be a valid USD prim")
    if instance_root_prim.IsInstanceProxy():
        raise RuntimeError(f"Cannot remap Action Graph paths on an instance proxy: {instance_root_prim.GetPath()}")

    source_stage = Usd.Stage.Open(source_usd_path)
    if source_stage is None:
        raise RuntimeError(f"Unable to open referenced Dynamic asset: {source_usd_path}")
    source_root = source_stage.GetDefaultPrim()
    if not source_root or not source_root.IsValid():
        raise RuntimeError(f"Referenced Dynamic asset must define a default prim: {source_usd_path}")
    source_root_path = source_root.GetPath()
    target_stage = instance_root_prim.GetStage()
    instance_root_path = instance_root_prim.GetPath()

    for prim in Usd.PrimRange(instance_root_prim):
        node_type = prim.GetAttribute("node:type")
        if not node_type or node_type.Get() != "omni.graph.scriptnode.ScriptNode":
            continue
        if not _is_action_graph_descendant(prim, instance_root_prim):
            continue
        for attribute in prim.GetAttributes():
            if not attribute.GetName().startswith("inputs:"):
                continue
            if attribute.GetTypeName() not in (Sdf.ValueTypeNames.String, Sdf.ValueTypeNames.Token):
                continue
            value = attribute.Get()
            if attribute.GetConnections() or not isinstance(value, str) or not value.startswith("/"):
                continue
            source_path = Sdf.Path(value)
            if not source_path.IsPrimPath() or not source_path.HasPrefix(source_root_path):
                continue
            target_path = source_path.ReplacePrefix(source_root_path, instance_root_path)
            if not (source_stage.GetPrimAtPath(source_path).IsValid() or target_stage.GetPrimAtPath(target_path).IsValid()):
                continue
            with Usd.EditContext(target_stage, target_stage.GetSessionLayer()):
                attribute.Set(target_path.pathString)
