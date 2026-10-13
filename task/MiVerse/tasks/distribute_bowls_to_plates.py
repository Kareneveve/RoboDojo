from itertools import combinations, permutations

from env.environment.task_env import TaskEnv
from task.MiVerse.task_common import MiVerseTaskCommon


class DistributeBowlsToPlatesCommon(MiVerseTaskCommon):
    step_limit = 2750

    bowl_labels = ("bowl_0", "bowl_1", "bowl_2")
    plate_labels = ("plate_north", "plate_east", "plate_west")
    chair_labels = ("chair_north", "chair_east", "chair_west")
    bowl_on_plate_max_z_above_plate = 0.05
    hand_clearance = 0.075

    instruction = (
        "Navigate to the dining table, unstack the three bowls one by one, "
        "and place one bowl upright onto each plate across the three place settings while avoiding collisions with the chairs."
    )

    def _hands_away_check(self, bowl_label):
        return self.reward_manager.is_no_ee_close_to_A(
            label=bowl_label,
            distance_threshold=self.hand_clearance,
        )

    def _bowl_on_plate_checks(self, bowl_label, plate_label):
        rm = self.reward_manager
        return [
            rm.is_not_moved(label=plate_label, dis_threshold=0.05),
            rm.is_A_in_B(label_A=bowl_label, label_B=plate_label),
            rm.is_A_z_lower_than_B_bbox_zmax(
                label_A=bowl_label,
                label_B=plate_label,
                z_threshold=self.bowl_on_plate_max_z_above_plate,
            ),
            rm.is_axis_up(label=bowl_label, axis=[0, 0, 1], threshold=15),
            rm.is_axis_up(label=plate_label, axis=[0, 0, 1], threshold=15),
            # Check release before updating the bowl's settled-pose cache.
            self._hands_away_check(bowl_label),
            rm.is_not_moved(label=bowl_label, dis_threshold=0.015, update=True),
        ]

    def _assignment_options(self):
        """Return the six valid one-bowl-per-plate final assignments."""
        return [
            [
                check
                for bowl_label, plate_label in zip(self.bowl_labels, plate_order)
                for check in self._bowl_on_plate_checks(bowl_label, plate_label)
            ]
            for plate_order in permutations(self.plate_labels)
        ]

    def _bowl_on_any_plate_options(self, bowl_label):
        return [self._bowl_on_plate_checks(bowl_label, plate_label) for plate_label in self.plate_labels]

    def _bowl_score_checks(self, bowl_label):
        return [self._bowl_on_any_plate_options(bowl_label)]

    def _single_item_score_options(self):
        return [self._bowl_score_checks(bowl_label) for bowl_label in self.bowl_labels]

    def _combined_item_score_options(self, count):
        return [
            [check for checks in selected_groups for check in checks]
            for selected_groups in combinations(self._single_item_score_options(), count)
        ]

    def _chair_moved_checks(self):
        rm = self.reward_manager
        return [rm.is_moved(label=chair_label, dis_threshold=0.02) for chair_label in self.chair_labels]

    def run_reward(self):
        # Any chair slide on any step fails (query count must stay 0).
        self.reward_manager.query([self._chair_moved_checks()], 0)
        self.reward_manager.check([self._assignment_options()])

    def get_score(self):
        rm = self.reward_manager
        rm.score(
            [
                [self._single_item_score_options()],
                [self._combined_item_score_options(2)],
                [*self._combined_item_score_options(3)[0]],
            ],
            [15, 40, 100],
            score_mode="transition",
        )


class distribute_bowls_to_plates(DistributeBowlsToPlatesCommon, TaskEnv):
    pass
