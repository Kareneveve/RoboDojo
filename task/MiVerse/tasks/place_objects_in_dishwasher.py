from env.environment.task_env import TaskEnv
from task.MiVerse.task_common import MiVerseTaskCommon


class PlaceObjectsInDishwasherCommon(MiVerseTaskCommon):
    step_limit = 2250

    target_label = "bowl"
    mug_label = "mug"
    dishwasher_label = "dishwasher"

    dishwasher_door_tag = "is_closed"
    dishwasher_door_open_ratio = 0.5

    dishwasher_rack_tags = ("rack001", "rack002", "rack003")
    dishwasher_rack_in_ratio = 0.9

    instruction = (
        "Pick up the bowl and mug from the kitchen counter, carry them to the dishwasher, "
        "open the dishwasher door, pull out either rack, place both items on the pulled-out rack, "
        "then push the rack back into the dishwasher."
    )

    def _bowl_in_dishwasher_check(self):
        return self.reward_manager.is_A_root_point_in_B_bbox(
            label_A=self.target_label,
            label_B=self.dishwasher_label,
            atol=-0.05,
        )

    def _mug_in_dishwasher_check(self):
        return self.reward_manager.is_A_root_point_in_B_bbox(
            label_A=self.mug_label,
            label_B=self.dishwasher_label,
            atol=-0.05,
        )

    def _items_in_dishwasher_checks(self):
        return [
            self._bowl_in_dishwasher_check(),
            self._mug_in_dishwasher_check(),
        ]

    def _door_open_check(self):
        return self.reward_manager.is_joint_position_above_ratio(
            label=self.dishwasher_label,
            percentage=self.dishwasher_door_open_ratio,
            tag=self.dishwasher_door_tag,
        )

    def _rack_in_check(self, tag):
        return self.reward_manager.is_joint_position_above_ratio(
            label=self.dishwasher_label,
            percentage=self.dishwasher_rack_in_ratio,
            tag=tag,
        )

    def _all_racks_in_checks(self):
        return [self._rack_in_check(tag) for tag in self.dishwasher_rack_tags]

    def _placed_and_racks_in_checks(self):
        return self._items_in_dishwasher_checks() + self._all_racks_in_checks()

    def run_reward(self):
        rm = self.reward_manager
        # Door may stay open. Success still requires it to have been opened.
        rm.check([self._door_open_check()])
        rm.check(self._placed_and_racks_in_checks())

    def get_score(self):
        rm = self.reward_manager
        rm.score(
            [
                [self._door_open_check()],
                self._items_in_dishwasher_checks(),
                self._placed_and_racks_in_checks(),
            ],
            [20, 50, 100],
            score_mode="transition",
        )


class place_objects_in_dishwasher(PlaceObjectsInDishwasherCommon, TaskEnv):
    pass
