from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.place_fruit_basket import PlaceFruitBasketCommon


class PlaceFruitBasketSpatialCommon(PlaceFruitBasketCommon):
    def _success_checks(self):
        return [
            self.reward_manager.is_A_root_point_in_world_circle(
                label_A=self.basket_label,
                center_xy=self.target_center_xy,
                radius=self.target_radius,
                z_bounds=self.target_z_bounds,
                room_object="coffee_table",
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
            room_objects=("coffee_table", "coffee_table", None),
        )


class place_fruit_basket_s(PlaceFruitBasketSpatialCommon, TaskEnv):
    pass
