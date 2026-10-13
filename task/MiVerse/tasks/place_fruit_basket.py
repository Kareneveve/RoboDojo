from env.environment.task_env import TaskEnv
from task.MiVerse.task_common import MiVerseTaskCommon


class PlaceFruitBasketCommon(MiVerseTaskCommon):
    step_limit = 1000
    basket_label = "fruit_basket"
    hand_clearance = 0.065

    # The TV is a free-standing articulation on TvStand051; any contact that
    # shifts or tilts it beyond these limits fails the episode.
    tv_label = "television"
    tv_translation_limit = 0.01
    tv_rotation_limit_deg = 3.0

    target_center_xy = (-0.4363248845, -0.0858775692)
    target_radius = 0.239358
    target_z_bounds = (0.5329411182, 0.5729411182)

    furniture_collision_bounds = (
        (
            (-0.4404155405, -0.4449682252, 0.0),
            (-0.4322342285, 0.2732130868, 0.4857311182),
        ),
        (
            (-0.7954155405, -0.0899682252, 0.0),
            (-0.0772342285, -0.0817869132, 0.4857311182),
        ),
        (
            (-2.0076433023, -0.9170379934, 0.0),
            (-1.6426438423, 0.8791682550, 0.4478819331),
        ),
    )

    instruction = (
        "Navigate to the TV stand, pick up the fruit basket to the left of the television, "
        "carry it over to the coffee table without bumping into either piece of furniture, and place it upright on the table."
    )

    def _success_checks(self):
        return [
            self.reward_manager.is_A_root_point_in_world_circle(
                label_A=self.basket_label,
                center_xy=self.target_center_xy,
                radius=self.target_radius,
                z_bounds=self.target_z_bounds,
            ),
            self.reward_manager.is_axis_up(
                label=self.basket_label,
                axis=[0.0, 0.0, 1.0],
                threshold=20,
            ),
            # Check release before updating the basket's settled-pose cache.
            self.reward_manager.is_no_ee_close_to_A(
                label=self.basket_label,
                distance_threshold=self.hand_clearance,
            ),
            self.reward_manager.is_not_moved(
                label=self.basket_label,
                dis_threshold=0.01,
                update=True,
            ),
        ]

    def _furniture_collision_check(self):
        return self.reward_manager.is_robot_bbox_intersecting_world_bbox(
            world_bounds=self.furniture_collision_bounds,
        )

    def _tv_touched_check(self):
        return self.reward_manager.is_pose_outside_initial_limits(
            label=self.tv_label,
            translation_threshold=self.tv_translation_limit,
            rotation_threshold_deg=self.tv_rotation_limit_deg,
        )

    def run_reward(self):
        # Base envelope must never enter the furniture keep-out boxes, and the
        # TV must never be displaced (proxy for the hands/arms touching it).
        self.reward_manager.query([self._furniture_collision_check()], 0)
        self.reward_manager.query([self._tv_touched_check()], 0)
        self.reward_manager.check(self._success_checks())

    def get_score(self):
        self.reward_manager.score([self._success_checks()], [100])


class place_fruit_basket(
    PlaceFruitBasketCommon,
    TaskEnv,
):
    pass
