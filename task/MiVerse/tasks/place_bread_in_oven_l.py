from env.description_manager.layout_instruction import LayoutInstructionMixin
from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.place_bread_in_oven import PlaceBreadInOvenCommon


class PlaceBreadInOvenLanguageCommon(LayoutInstructionMixin, PlaceBreadInOvenCommon):
    pass


class place_bread_in_oven_l(PlaceBreadInOvenLanguageCommon, TaskEnv):
    pass
