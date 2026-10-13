from env.environment.task_env import TaskEnv
from task.MiVerse.task_common import MiVerseTaskCommon


class SweepPaperBallIntoDustbinCommon(MiVerseTaskCommon):
    step_limit = 1850

    paper_ball_labels = (
        "paper_ball_1",
        "paper_ball_2",
        "paper_ball_3",
        "paper_ball_4",
    )
    dustbin_label = "dustbin"
    dustbin_bbox_margin = -0.015

    instruction = (
        "Navigate to the left side of the TV stand, pick up the dustbin, carry it next to the coffee table, "
        "and sweep all the paper balls from the table into the dustbin."
    )

    def _ball_in_dustbin_check(self, ball_label):
        return [
            self.reward_manager.is_A_bbox_in_B_bbox(
                label_A=ball_label,
                label_B=self.dustbin_label,
                margin=self.dustbin_bbox_margin,
            ),
        ]

    def _success_checks(self):
        checks = []
        for ball_label in self.paper_ball_labels:
            checks.extend(self._ball_in_dustbin_check(ball_label))
        return checks

    def run_reward(self):
        # Every paper ball ends up in the dustbin.
        self.reward_manager.check(self._success_checks())

    def get_score(self):
        # Progressive credit per ball collected; 100 only when all four are in.
        self.reward_manager.score(
            [self._ball_in_dustbin_check(ball_label) for ball_label in self.paper_ball_labels],
            [25, 50, 75, 100],
            score_mode="by_count",
        )


class sweep_paper_ball_into_dustbin(SweepPaperBallIntoDustbinCommon, TaskEnv):
    pass
