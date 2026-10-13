"""Shared lifecycle for MiVerse evaluation tasks."""

from env.reward_manager.reward_manager import RewardManager


class MiVerseTaskCommon:
    instruction = ""
    step_limit = 2000

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = self.step_limit

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.reward_manager.reset()
        return result

    def gen_instruction(self, env_idx):
        return [self.instruction]
