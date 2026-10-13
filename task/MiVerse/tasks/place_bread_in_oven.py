from env.environment.task_env import TaskEnv
from task.MiVerse.task_common import MiVerseTaskCommon


class PlaceBreadInOvenCommon(MiVerseTaskCommon):
    step_limit = 1750

    target_label = "bread"
    oven_label = "oven"

    oven_door_tag = "is_closed"
    oven_door_closed_ratio = 0.1
    oven_door_open_ratio = 0.7

    oven_world_bounds = (
        (-3.507498612, -1.97472983, 0.940084503),
        (-3.091498612, -1.442472983, 1.195084503),
    )

    instruction = (
        "Pick up the bread from the kitchen counter, "
        "navigate to the oven, open the oven door, place the bread inside the oven, and then close the oven door."
    )

    def _object_in_oven_check(self):
        return self.reward_manager.is_A_root_point_in_world_bbox(
            label_A=self.target_label,
            world_bounds=self.oven_world_bounds,
        )

    def _bread_settled_check(self):
        return self.reward_manager.is_not_moved(
            label=self.target_label,
            dis_threshold=0.01,
            update=True,
        )

    def _oven_door_open_check(self):
        return self.reward_manager.is_joint_position_above_ratio(
            label=self.oven_label,
            percentage=self.oven_door_open_ratio,
            tag=self.oven_door_tag,
        )

    def _oven_door_closed_check(self):
        return self.reward_manager.is_joint_position_below_ratio(
            label=self.oven_label,
            percentage=self.oven_door_closed_ratio,
            tag=self.oven_door_tag,
        )

    def _bread_inside_while_door_open_checks(self):
        # Settled must stay last: check evaluation short-circuits, so the
        # cached pose only advances once the bread is already in the cavity
        # with the door still open.
        return [
            self._object_in_oven_check(),
            self._oven_door_open_check(),
            self._bread_settled_check(),
        ]

    def _success_checks(self):
        return [
            self._object_in_oven_check(),
            self._oven_door_closed_check(),
            self._bread_settled_check(),
        ]

    def run_reward(self):
        self.reward_manager.check([self._oven_door_open_check()])
        # Reject using the closing door as a pusher: bread must already rest
        # inside the cavity while the door is still open.
        self.reward_manager.check(self._bread_inside_while_door_open_checks())
        self.reward_manager.check(self._success_checks())

    def get_score(self):
        rm = self.reward_manager
        rm.score(
            [
                [self._oven_door_open_check()],
                self._bread_inside_while_door_open_checks(),
                self._success_checks(),
            ],
            [20, 50, 100],
            score_mode="transition",
        )


class place_bread_in_oven(PlaceBreadInOvenCommon, TaskEnv):
    pass
