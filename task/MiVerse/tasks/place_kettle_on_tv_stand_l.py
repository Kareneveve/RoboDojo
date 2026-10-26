from env.description_manager.layout_instruction import LayoutInstructionMixin
from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.place_kettle_on_tv_stand import PlaceKettleOnTvStandCommon


class PlaceKettleOnTvStandLanguageCommon(LayoutInstructionMixin, PlaceKettleOnTvStandCommon):
    pass


class place_kettle_on_tv_stand_l(PlaceKettleOnTvStandLanguageCommon, TaskEnv):
    pass
