from env.description_manager.layout_instruction import LayoutInstructionMixin
from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.distribute_bowls_to_plates import DistributeBowlsToPlatesCommon


class DistributeBowlsToPlatesLanguageCommon(LayoutInstructionMixin, DistributeBowlsToPlatesCommon):
    pass


class distribute_bowls_to_plates_l(DistributeBowlsToPlatesLanguageCommon, TaskEnv):
    pass
