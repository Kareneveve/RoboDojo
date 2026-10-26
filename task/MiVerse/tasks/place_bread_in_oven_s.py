from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.place_bread_in_oven import PlaceBreadInOvenCommon


class PlaceBreadInOvenSpatialCommon(PlaceBreadInOvenCommon):
    step_limit = 1900

    refresh_instruction_on_reset = True

    instruction = (
        "Pick up the bread from the {source}, "
        "navigate to the oven, open the oven door, place the bread inside the oven, and then close the oven door."
    )

    def gen_instruction(self, env_idx):
        manager = self.scene_manager.layout_manager
        layout = self.scene_manager.config.get(f"env{env_idx}")
        if layout is None:
            # Failed placements are skipped; initialization still asks for text.
            return [self.instruction.format(source="kitchen island, cutting board, or dining table")]

        targets = [
            obj
            for instances in layout.get("Rigid", {}).values()
            for obj in instances
            if obj.get("label") == self.target_label
        ]
        if len(targets) != 1:
            raise ValueError(f"Expected one {self.target_label!r} in scene layout for env {env_idx}")
        target = targets[0]
        source_names = {
            "kitchen_island": "kitchen island",
            "cutting_board": "cutting board",
            "table": "dining table",
        }
        plane = target.get("relative_plane")
        if plane not in source_names:
            raise ValueError(f"Unknown source surface {plane!r} for env {env_idx}")
        category = target.get("category")
        if category != "bread":
            raise ValueError(f"Unknown target category {category!r} for env {env_idx}")
        source = source_names[plane]
        instruction = self.instruction.format(source=source)
        if manager.replay and "instruction" in layout:
            saved = layout["instruction"]
            if not isinstance(saved, str) or not saved.strip():
                raise ValueError(f"Scene layout for env {env_idx} has an invalid instruction")
            # Preserve saved wording when it names the actual pickup surface.
            # Older generic or mismatched text is rebuilt from this env's layout.
            if saved.strip().casefold().startswith(f"Pick up the bread from the {source},".casefold()):
                return [saved]

        layout["instruction"] = instruction
        return [instruction]


class place_bread_in_oven_s(PlaceBreadInOvenSpatialCommon, TaskEnv):
    pass
