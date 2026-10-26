from env.description_manager.layout_instruction import LayoutInstructionMixin
from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.sweep_paper_ball_into_dustbin import SweepPaperBallIntoDustbinCommon


class SweepPaperBallIntoDustbinLanguageCommon(LayoutInstructionMixin, SweepPaperBallIntoDustbinCommon):
    pass


class sweep_paper_ball_into_dustbin_l(SweepPaperBallIntoDustbinLanguageCommon, TaskEnv):
    pass
