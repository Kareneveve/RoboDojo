from env.description_manager.layout_instruction import LayoutInstructionMixin
from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.put_object_in_sink import PutObjectInSinkCommon


class PutObjectInSinkLanguageCommon(LayoutInstructionMixin, PutObjectInSinkCommon):
    pass


class put_object_in_sink_l(PutObjectInSinkLanguageCommon, TaskEnv):
    pass
