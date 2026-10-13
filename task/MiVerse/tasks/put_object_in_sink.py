from env.environment.task_env import TaskEnv
from task.MiVerse.task_common import MiVerseTaskCommon


class PutObjectInSinkCommon(MiVerseTaskCommon):
    step_limit = 1600
    target_label = "target"
    target_settled_limit = 0.02
    # Wrist-to-object-bbox clearance. Dex hands have no open/close scalar;
    # the EE must leave the object's bbox by this margin after release.
    hand_clearance = 0.08
    cooker_label = "electric_cooker"
    cooker_translation_limit = 0.05
    cooker_rotation_limit_deg = 10.0

    # World-aligned bounds authored by Sink001's reg_basin_left and
    # reg_basin_right semantic sites in Kitchen009. Sink001 belongs to the
    # read-only Room USD and is not a normal LayoutManager object label.
    sink_world_bounds = (
        (
            (-2.522815, 1.573674, 0.691976),
            (-2.261801, 1.895704, 0.836198),
        ),
        (
            (-2.206664, 1.567471, 0.693070),
            (-1.858459, 1.904166, 0.836198),
        ),
    )

    instruction = (
        "Find either a mug or a bowl on the kitchen counter, dining table, "
        "or cutting board, navigate to the item, pick it up, move to the sink, and place it inside the sink."
    )

    def _sink_check(self):
        return self.reward_manager.is_A_root_point_in_world_bbox(
            label_A=self.target_label,
            world_bounds=self.sink_world_bounds,
        )

    def _hands_away_check(self):
        return self.reward_manager.is_no_ee_close_to_A(
            label=self.target_label,
            distance_threshold=self.hand_clearance,
        )

    def _target_settled_check(self):
        return self.reward_manager.is_not_moved(
            label=self.target_label,
            dis_threshold=self.target_settled_limit,
            update=True,
        )

    def _cooker_pose_check(self):
        return self.reward_manager.is_pose_within_initial_limits(
            label=self.cooker_label,
            translation_threshold=self.cooker_translation_limit,
            rotation_threshold_deg=self.cooker_rotation_limit_deg,
        )

    def _cooker_pose_violation(self):
        return self.reward_manager.is_pose_outside_initial_limits(
            label=self.cooker_label,
            translation_threshold=self.cooker_translation_limit,
            rotation_threshold_deg=self.cooker_rotation_limit_deg,
        )

    def _success_checks(self):
        # Hands-away must come before settled: while a wrist is still next
        # to the object, check() short-circuits and the pose cache does not
        # follow the held item. Settled stays last.
        return [
            self._sink_check(),
            self._cooker_pose_check(),
            self._hands_away_check(),
            self._target_settled_check(),
        ]

    def run_reward(self):
        self.reward_manager.query([self._cooker_pose_violation()], 0)
        self.reward_manager.check(self._success_checks())

    def get_score(self):
        self.reward_manager.score([self._success_checks()], [100])


class put_object_in_sink(PutObjectInSinkCommon, TaskEnv):
    pass
