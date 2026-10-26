from env.environment.task_env import TaskEnv
from task.MiVerse.tasks.put_object_in_sink import PutObjectInSinkCommon


class PutObjectInSinkSpatialCommon(PutObjectInSinkCommon):
    step_limit = 1800


class put_object_in_sink_s(PutObjectInSinkSpatialCommon, TaskEnv):
    pass
