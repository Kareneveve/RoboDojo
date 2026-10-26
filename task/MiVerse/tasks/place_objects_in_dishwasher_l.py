from env.description_manager.layout_instruction import LayoutInstructionMixin
from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.place_objects_in_dishwasher import PlaceObjectsInDishwasherCommon


class PlaceObjectsInDishwasherLanguageCommon(LayoutInstructionMixin, PlaceObjectsInDishwasherCommon):
    pass


class place_objects_in_dishwasher_l(PlaceObjectsInDishwasherLanguageCommon, TaskEnv):
    pass
