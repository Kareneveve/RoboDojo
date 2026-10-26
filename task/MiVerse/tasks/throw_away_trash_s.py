from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.throw_away_trash import ThrowAwayTrashCommon


class ThrowAwayTrashSpatialCommon(ThrowAwayTrashCommon):
    step_limit = 1250

    # Each replay batch can assign a different package category to this slot.
    refresh_instruction_on_reset = True


class throw_away_trash_s(ThrowAwayTrashSpatialCommon, TaskEnv):
    pass
