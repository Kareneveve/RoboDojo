from env.description_manager.layout_instruction import LayoutInstructionMixin
from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.place_fruit_basket import PlaceFruitBasketCommon


class PlaceFruitBasketLanguageCommon(LayoutInstructionMixin, PlaceFruitBasketCommon):
    pass


class place_fruit_basket_l(PlaceFruitBasketLanguageCommon, TaskEnv):
    pass
