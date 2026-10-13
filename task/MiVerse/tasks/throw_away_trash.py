from env.environment.task_env import TaskEnv
from task.MiVerse.task_common import MiVerseTaskCommon


class ThrowAwayTrashCommon(MiVerseTaskCommon):
    step_limit = 1150
    package_label = "discarded_package"
    dustbin_label = "dustbin"
    robot_vacuum_label = "robot_vacuum_obstacle"

    instruction = (
        "Pick up the package from the table, carry it out of the room to the dustbin, "
        "and place it completely inside the dustbin while avoiding the "
        "moving robot vacuum at all times."
    )
    _package_nouns = {
        "milk_box": "milk carton",
        "bottle": "bottle",
    }

    def _discarded_package_category(self, env_idx):
        layout = self.scene_manager.layout_manager
        for obj_cfg in layout.get_layout_records(env_idx, "Rigid"):
            if obj_cfg.get("label") == self.package_label:
                return obj_cfg.get("category")
        return None

    def gen_instruction(self, env_idx):
        category = self._discarded_package_category(env_idx)
        package_noun = self._package_nouns.get(category, "package")
        return [self.instruction.replace("package", package_noun)]

    def _package_in_dustbin_check(self):
        return self.reward_manager.is_A_bbox_in_B_bbox(
            label_A=self.package_label,
            label_B=self.dustbin_label,
        )

    def _robot_vacuum_collision_check(self):
        return self.reward_manager.is_robot_bbox_intersecting_layout_object(
            label=self.robot_vacuum_label,
            robot_index=0,
        )

    def run_reward(self):
        self.reward_manager.query([self._robot_vacuum_collision_check()], 0)
        self.reward_manager.check([self._package_in_dustbin_check()])

    def get_score(self):
        self.reward_manager.score([[self._package_in_dustbin_check()]], [100])


class throw_away_trash(ThrowAwayTrashCommon, TaskEnv):
    pass
