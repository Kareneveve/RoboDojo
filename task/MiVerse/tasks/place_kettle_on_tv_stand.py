from env.environment.task_env import TaskEnv
from task.MiVerse.task_common import MiVerseTaskCommon


class PlaceKettleOnTvStandCommon(MiVerseTaskCommon):
    step_limit = 1000
    kettle_label = "electric_kettle"
    target_world_bounds = (
        (-1.932261175, -0.902372833, 0.562),
        (-1.662261175, -0.632372833, 0.582),
    )

    # The TV is a free-standing articulation on TvStand051; any contact that
    # shifts or tilts it beyond these limits fails the episode.
    tv_label = "television"
    tv_translation_limit = 0.01
    tv_rotation_limit_deg = 3.0

    instruction = (
        "Navigate to the table, pick up the kettle, "
        "carry it to the TV stand without bumping into either piece of furniture, "
        "and place it upright to the left of the television."
    )

    def _success_checks(self):
        return [
            self.reward_manager.is_A_root_point_in_world_bbox(
                label_A=self.kettle_label,
                world_bounds=self.target_world_bounds,
            ),
            self.reward_manager.is_axis_up(
                label=self.kettle_label,
                axis=[0.0, 0.0, 1.0],
                threshold=10,
            ),
            # Settled must stay last: check evaluation short-circuits, so the
            # cached pose only advances once the kettle is already in bounds.
            self.reward_manager.is_not_moved(
                label=self.kettle_label,
                dis_threshold=0.01,
                update=True,
            ),
        ]

    def _tv_touched_check(self):
        return self.reward_manager.is_pose_outside_initial_limits(
            label=self.tv_label,
            translation_threshold=self.tv_translation_limit,
            rotation_threshold_deg=self.tv_rotation_limit_deg,
        )

    def run_reward(self):
        # The TV must never be displaced (proxy for the hands/arms touching it).
        self.reward_manager.query([self._tv_touched_check()], 0)
        self.reward_manager.check(self._success_checks())

    def get_score(self):
        self.reward_manager.score([self._success_checks()], [100])


class place_kettle_on_tv_stand(
    PlaceKettleOnTvStandCommon,
    TaskEnv,
):
    pass
