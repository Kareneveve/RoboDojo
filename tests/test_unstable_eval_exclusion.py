"""Verify unstable episodes are excluded from eval metrics, details, and videos."""

from __future__ import annotations

import importlib
import types

from omegaconf import OmegaConf
import pytest

from utils.cluttered_generator import UnStableError


def _make_eval_env(tmp_path, monkeypatch, num_envs: int, layout_valid=None, stability_result=(True, [])):
    from src.eval_client import eval_env as eval_env_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROBODOJO_RUN_ID", "test_unstable_run")

    if layout_valid is None:
        layout_valid = [True] * num_envs

    class _StubRewardManager:
        def get_reward(self, final_check=False, env_idx_list=None):
            return [0.0] * num_envs

        def get_score(self, reward_lst=None):
            if reward_lst is None:
                return [0.0] * num_envs
            return [100.0 if r > 0.5 else 25.0 for r in reward_lst]

    class _StubLayoutManager:
        def __init__(self):
            self.replay = False
            self.layout_valid = list(layout_valid)

        def check_layout_stability(self, env):
            return stability_result

    class _StubSceneManager:
        def __init__(self):
            self.layout_manager = _StubLayoutManager()

        def apply_saved_poses(self, env_idx_list=None):
            pass

    class _StubTask:
        def __init__(self, config, app, **kwargs):
            self.num_envs = num_envs
            self.dt = 0.004
            self.env_seed_list = list(range(num_envs))
            self.step_lim = 5
            self.reward_manager = _StubRewardManager()
            self.robot_manager = types.SimpleNamespace(robot_list=[])
            self.scene_manager = _StubSceneManager()
            self.contact_monitor = None

        def run_reward(self):
            pass

        def get_score(self):
            pass

        def render(self):
            pass

        def sim_step(self, *args, **kwargs):
            pass

    class _FakeRegistry:
        @staticmethod
        def load_task_class(name):
            return name, _StubTask

    orig_import = importlib.import_module

    def _patched_import(name, *args, **kwargs):
        if name.endswith(".task_registry"):
            return _FakeRegistry
        return orig_import(name, *args, **kwargs)

    class _Noop:
        def __init__(self, *a, **k):
            pass

        def get_seeds(self, *a, **k):
            return list(range(num_envs))

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(importlib, "import_module", _patched_import)
    monkeypatch.setattr(eval_env_module, "ObsManager", _Noop)
    monkeypatch.setattr(eval_env_module, "SeedManager", _Noop)
    monkeypatch.setattr(eval_env_module, "WsModelClient", _Noop)
    monkeypatch.setattr(eval_env_module, "get_robot_action_dim_info", lambda *a, **k: {"arm_dim": [6], "ee_dim": [1]})

    cfg = OmegaConf.create(
        {
            "eval_cfg": {
                "task_name": "play_tic_tac_toe",
                "config_name": "arx_x5",
                "policy_name": "dummy",
                "seed": 0,
                "eval_num": 10,
                "num_envs": num_envs,
            },
            "deploy_cfg": {"host": "127.0.0.1", "port": 1, "protocol": "ws"},
            "robot": {},
        }
    )
    env = eval_env_module.create_eval_env(cfg, app=None)
    env.env_seeds = [100 + i for i in range(num_envs)]
    env.current_env_seed_map = {i: 100 + i for i in range(num_envs)}
    env.episode_nums = num_envs
    env.success = [True] * num_envs
    env.end_flag = [False] * num_envs
    env.unstable_envs = set()
    return env


def test_mid_episode_unstable_excluded_from_metrics_details_and_video(tmp_path, monkeypatch):
    env = _make_eval_env(tmp_path, monkeypatch, num_envs=2)
    saved_videos = []
    aborted_envs = []
    env.save_video = lambda env_idx, path, tag: saved_videos.append((env_idx, tag))
    orig_abort = env._abort_video_writers

    def _tracked_abort(env_idx_list=None):
        aborted_envs.append(env_idx_list)
        orig_abort(env_idx_list=env_idx_list)

    env._abort_video_writers = _tracked_abort

    def _run_episode():
        env.success[0] = True
        env.end_flag[0] = True
        env.mark_env_unstable(1)
        # mark_env_unstable must immediately stop stepping that env
        assert env.end_flag[1] is True
        assert env.success[1] is False

    env.eval_batch = True
    env.eval_one_episode_batch = _run_episode

    env.run_eval()

    assert [1] in aborted_envs
    assert env.success_nums == 1
    assert env.fail_nums == 0
    assert env.unstable_nums == 1
    assert env.unstable_seeds == {101}
    assert env.eval_result["eval_time"] == 1
    assert env.eval_result["success_rate"] == pytest.approx(1.0)
    assert env.eval_result["score"] == pytest.approx(100.0)
    assert list(env.eval_result["details"].keys()) == [0]
    assert env.eval_result["details"][0]["layout_id"] == 100
    assert saved_videos == [(0, "success")]


def test_setup_scene_partial_unstable_and_invalid_layout_excluded(tmp_path, monkeypatch):
    env = _make_eval_env(
        tmp_path,
        monkeypatch,
        num_envs=3,
        layout_valid=[True, False, True],
        stability_result=(True, [2]),
    )
    saved_videos = []
    env.save_video = lambda env_idx, path, tag: saved_videos.append((env_idx, tag))

    env.setup_scene()
    assert env.unstable_nums == 1
    assert env.episode_nums == 1
    assert env.end_flag == [False, True, True]

    def _run_episode():
        env.success[0] = True
        env.end_flag[0] = True

    env.eval_batch = True
    env.eval_one_episode_batch = _run_episode

    env.run_eval()

    assert env.success_nums == 1
    assert env.fail_nums == 0
    assert env.unstable_nums == 1  # not double-counted
    assert env.eval_result["eval_time"] == 1
    assert env.eval_result["success_rate"] == pytest.approx(1.0)
    assert env.eval_result["score"] == pytest.approx(100.0)
    assert list(env.eval_result["details"].keys()) == [0]
    assert env.eval_result["details"][0]["layout_id"] == 100
    assert saved_videos == [(0, "success")]


def test_setup_scene_all_unstable_raises_unstable_error(tmp_path, monkeypatch):
    env = _make_eval_env(
        tmp_path,
        monkeypatch,
        num_envs=2,
        layout_valid=[True, True],
        stability_result=(False, [0, 1]),
    )

    with pytest.raises(UnStableError):
        env.setup_scene()

    assert env.unstable_nums == 2
    assert env.episode_nums == 0
    assert env.fail_nums == 0
    assert env.eval_result["details"] == {}


def test_main_unstable_error_does_not_record_batch_failure(monkeypatch):
    from tests.test_p1_fixes import _load_main_module_stubbed

    events = []

    class _FakeMonitor:
        def start(self, enabled=True):
            pass

        def reset(self):
            pass

        def is_fatal(self):
            return False

        def get_broken_envs(self):
            return set()

        def shutdown(self):
            pass

    class _FakeEnv:
        num_envs = 2
        success_nums = 0
        fail_nums = 0
        unstable_nums = 0
        current_env_seed_map = {0: 0, 1: 1}
        env_seeds = None

        def __init__(self):
            self._batch = 0
            self.model_client = types.SimpleNamespace(close=lambda: None)
            self.seed_manager = types.SimpleNamespace(
                get_seeds=self._get_seeds,
                eval_step=lambda: events.append("eval_step"),
            )

        def _get_seeds(self, max_count=2):
            self._batch += 1
            if self._batch == 1:
                return [0, 1]
            if self._batch == 2:
                return [2, 3]
            return None

        def reset(self, seed=None):
            if seed == [0, 1]:
                self.unstable_nums += 2
                raise UnStableError("All scene Unstable Error!")

        def run_eval(self):
            self.success_nums += 2

        def record_batch_failure(self, err, env_idxs=None):
            events.append("record_batch_failure")

        def close(self):
            pass

    env_obj = _FakeEnv()
    sim_app = types.SimpleNamespace(close=lambda: None)
    mod, _ = _load_main_module_stubbed(monkeypatch, monitor=_FakeMonitor(), env_obj=env_obj, sim_app=sim_app)
    monkeypatch.setattr(mod, "_delete_resume_manifest", lambda env: None)

    mod.main()

    assert "record_batch_failure" not in events
    assert events == ["eval_step", "eval_step"]
    assert env_obj.success_nums == 2
    assert env_obj.fail_nums == 0
    assert env_obj.unstable_nums == 2


def test_main_physx_broken_abandons_and_refills_without_counting_failure(monkeypatch):
    from tests.test_p1_fixes import _load_main_module_stubbed

    events = []
    reset_batches = []

    class _FakeMonitor:
        def start(self, enabled=True):
            pass

        def reset(self):
            pass

        def is_fatal(self):
            return False

        def get_broken_envs(self):
            return set()

        def shutdown(self):
            pass

    class _FakeEnv:
        num_envs = 2
        success_nums = 0
        fail_nums = 0
        unstable_nums = 0
        current_env_seed_map = {}
        env_seeds = None

        def __init__(self):
            self._queue = [0, 1, 2]
            self._broken_once = False
            self.abandoned_seeds = set()
            self.model_client = types.SimpleNamespace(close=lambda: None)
            self.seed_manager = types.SimpleNamespace(
                get_seeds=self._get_seeds,
                eval_step=lambda: events.append("eval_step"),
            )

        def _get_seeds(self, max_count=2):
            if not self._queue:
                return None
            n = min(max_count, len(self._queue))
            batch = self._queue[:n]
            self._queue = self._queue[n:]
            return batch

        def get_seeds_for_envs(self, env_idxs):
            return {self.current_env_seed_map[i] for i in env_idxs if i in self.current_env_seed_map}

        def persist_resume_manifest(self):
            events.append(("persist", sorted(self.abandoned_seeds)))

        def reset(self, seed=None):
            reset_batches.append(list(seed))
            self.current_env_seed_map = {i: s for i, s in enumerate(seed) if s is not None}

        def run_eval(self):
            if not self._broken_once:
                self._broken_once = True
                raise mod.PhysXBrokenError({0})
            self.success_nums += len(self.current_env_seed_map)

        def record_batch_failure(self, err, env_idxs=None):
            events.append("record_batch_failure")

        def close(self):
            pass

    env_obj = _FakeEnv()
    sim_app = types.SimpleNamespace(close=lambda: None)
    mod, _ = _load_main_module_stubbed(monkeypatch, monitor=_FakeMonitor(), env_obj=env_obj, sim_app=sim_app)
    monkeypatch.setattr(mod, "_delete_resume_manifest", lambda env: None)

    mod.main()

    assert "record_batch_failure" not in events
    assert env_obj.abandoned_seeds == {0}
    assert reset_batches == [[0, 1], [2, 1]]
    assert env_obj.success_nums == 2
    assert env_obj.fail_nums == 0


def test_record_batch_failure_skips_unstable_envs(tmp_path, monkeypatch):
    env = _make_eval_env(tmp_path, monkeypatch, num_envs=2)
    env.mark_env_unstable(0)

    env.record_batch_failure(RuntimeError("policy crash"))

    assert env.fail_nums == 1
    assert list(env.eval_result["details"].keys()) == [0]
    assert env.eval_result["details"][0]["layout_id"] == 101


def test_resume_restores_unstable_nums_and_filters_unstable_layouts(tmp_path, monkeypatch):
    import json

    from src.eval_client import eval_env as eval_env_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROBODOJO_RUN_ID", "test_resume_unstable")

    init_eval_calls = []

    class _StubSeedManager:
        def __init__(self, *a, **k):
            pass

        def init_eval(self, completed_layout_ids=None, abandoned_layout_ids=None, unstable_layout_ids=None):
            init_eval_calls.append(
                {
                    "completed": list(completed_layout_ids or []),
                    "abandoned": list(abandoned_layout_ids or []),
                    "unstable": list(unstable_layout_ids or []),
                }
            )

    class _StubTask:
        def __init__(self, config, app, **kwargs):
            self.num_envs = 2
            self.dt = 0.004
            self.env_seed_list = [0, 1]
            self.step_lim = 5
            self.reward_manager = types.SimpleNamespace()
            self.robot_manager = types.SimpleNamespace(robot_list=[])
            self.scene_manager = types.SimpleNamespace(layout_manager=types.SimpleNamespace(replay=False))
            self.contact_monitor = None

    class _FakeRegistry:
        @staticmethod
        def load_task_class(name):
            return name, _StubTask

    orig_import = importlib.import_module

    def _patched_import(name, *args, **kwargs):
        if name.endswith(".task_registry"):
            return _FakeRegistry
        return orig_import(name, *args, **kwargs)

    class _Noop:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(importlib, "import_module", _patched_import)
    monkeypatch.setattr(eval_env_module, "ObsManager", _Noop)
    monkeypatch.setattr(eval_env_module, "SeedManager", _StubSeedManager)
    monkeypatch.setattr(eval_env_module, "WsModelClient", _Noop)
    monkeypatch.setattr(eval_env_module, "get_robot_action_dim_info", lambda *a, **k: {"arm_dim": [6], "ee_dim": [1]})

    cfg = OmegaConf.create(
        {
            "eval_cfg": {
                "task_name": "play_tic_tac_toe",
                "config_name": "arx_x5",
                "policy_name": "dummy",
                "seed": 0,
                "eval_num": 10,
                "num_envs": 2,
            },
            "deploy_cfg": {"host": "127.0.0.1", "port": 1, "protocol": "ws"},
            "robot": {},
        }
    )

    resume_state = {
        "success_nums": 1,
        "fail_nums": 0,
        "unstable_nums": 3,
        "total_score": 1.0,
        "abandoned_layout_ids": [5],
        "unstable_layout_ids": [2, 4, 6],
        "details": {"0": {"layout_id": 0, "success": True, "score": 1.0}},
    }

    env = eval_env_module.create_eval_env(cfg, app=None, resume_state=resume_state)

    assert env.unstable_nums == 3
    assert env.unstable_seeds == {2, 4, 6}
    assert env.abandoned_seeds == {5}
    assert init_eval_calls == [{"completed": [0], "abandoned": [5], "unstable": [2, 4, 6]}]

    manifest_path = env.persist_resume_manifest()
    with open(manifest_path, encoding="utf-8") as fp:
        persisted = json.load(fp)
    assert persisted["unstable_nums"] == 3
    assert persisted["unstable_layout_ids"] == [2, 4, 6]
    assert persisted["abandoned_layout_ids"] == [5]


def test_reset_does_not_capture_duplicate_initial_obs_frame(tmp_path, monkeypatch):
    env = _make_eval_env(tmp_path, monkeypatch, num_envs=2)
    get_obs_calls = []
    env.get_obs_batch = lambda *a, **k: get_obs_calls.append((a, k))
    env.setup_scene = lambda: None
    env.obs_manager = types.SimpleNamespace(reset=lambda: None)
    env.robot_manager = types.SimpleNamespace(
        robot_list=[],
        set_origin_endpose=lambda: None,
        set_robot_init_state=lambda: None,
    )
    env.reward_manager = types.SimpleNamespace(init_state=lambda: None)
    env.scene_manager.layout_manager.set_saved_layout = lambda idx, info: None
    env.seed_manager.get_seed_scene_info = lambda seed: {}

    # Stub BaseEnv.reset on the base class
    env.__class__.__mro__[1].reset = lambda self, seed=None, options=None: None

    result = env.reset(seed=[10, 11])
    assert result is None
    assert get_obs_calls == []


def test_physx_retry_does_not_rerun_already_unstable_seed(monkeypatch):
    from tests.test_p1_fixes import _load_main_module_stubbed

    reset_batches = []

    class _FakeMonitor:
        def start(self, enabled=True):
            pass

        def reset(self):
            pass

        def is_fatal(self):
            return False

        def get_broken_envs(self):
            return set()

        def shutdown(self):
            pass

    class _FakeEnv:
        num_envs = 2
        success_nums = 0
        fail_nums = 0
        unstable_nums = 0
        current_env_seed_map = {}
        env_seeds = None

        def __init__(self):
            self._queue = [0, 1, 2, 3]
            self._broken_once = False
            self.abandoned_seeds = set()
            self.unstable_seeds = set()
            self.model_client = types.SimpleNamespace(close=lambda: None)
            self.seed_manager = types.SimpleNamespace(
                get_seeds=self._get_seeds,
                eval_step=lambda: None,
            )

        def _get_seeds(self, max_count=2):
            if not self._queue:
                return None
            n = min(max_count, len(self._queue))
            batch = self._queue[:n]
            self._queue = self._queue[n:]
            return batch

        def get_seeds_for_envs(self, env_idxs):
            return {self.current_env_seed_map[i] for i in env_idxs if i in self.current_env_seed_map}

        def persist_resume_manifest(self):
            pass

        def reset(self, seed=None):
            reset_batches.append(list(seed))
            self.current_env_seed_map = {i: s for i, s in enumerate(seed) if s is not None}
            # Simulate seed 1 failing layout stability during setup_scene on round 1
            if 1 in self.current_env_seed_map.values():
                self.unstable_nums += 1
                self.unstable_seeds.add(1)

        def run_eval(self):
            if not self._broken_once:
                self._broken_once = True
                raise mod.PhysXBrokenError({0})
            self.success_nums += len(self.current_env_seed_map)

        def close(self):
            pass

    env_obj = _FakeEnv()
    sim_app = types.SimpleNamespace(close=lambda: None)
    mod, _ = _load_main_module_stubbed(monkeypatch, monitor=_FakeMonitor(), env_obj=env_obj, sim_app=sim_app)
    monkeypatch.setattr(mod, "_delete_resume_manifest", lambda env: None)

    mod.main()

    # Seed 0 abandoned & replaced with 2; seed 1 (already unstable) NOT re-run; then seed 3 fills eval_num=2
    assert reset_batches == [[0, 1], [2, None], [3]]
    assert env_obj.unstable_nums == 1
    assert env_obj.abandoned_seeds == {0}
    assert env_obj.success_nums == 2


