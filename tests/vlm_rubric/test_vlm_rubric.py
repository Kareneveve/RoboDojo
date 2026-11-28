import base64
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from PIL import Image

from src.vlm_rubric import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RUBRIC_REPORT_JSON,
    SIM_BACKUP_FILENAME,
    TASK_RUBRICS,
    VLMJudgeClient,
    VLMJudgeError,
    apply_rubric_verdict,
    configure_rubric_logging,
    evaluate_run,
    extract_criterion_ids,
    get_rubric_criteria_prompt,
    get_task_rubric,
    restore_sim_result,
)
from src.vlm_rubric.judge import _json_from_text, _message_content, _normalize_judge_output
from src.vlm_rubric.media import (
    CAMERAS,
    EpisodeMedia,
    discover_episodes,
    load_result,
    stitch_camera_frames,
    synchronized_frames,
)


def _make_jpeg(color: tuple[int, int, int] = (128, 128, 128), size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _three_view_frames(count: int = 2) -> dict[str, list[bytes]]:
    colors = {"head": (220, 40, 40), "left_wrist": (40, 220, 40), "right_wrist": (40, 40, 220)}
    return {camera: [_make_jpeg(colors[camera]) for _ in range(count)] for camera in CAMERAS}


class FailJudgeClient:
    model = "qwen3.8-max"

    def complete(self, content, *, system_prompt):
        self.content = content
        self.system_prompt = system_prompt
        return json.dumps(
            {
                "verdict": "fail",
                "confidence": 0.9,
                "violation_detected": True,
                "criteria": [{"id": "no_sweep", "met": False}],
                "evidence": [{"frame_index": 1, "time_s": 0.5, "cameras": list(CAMERAS), "description": "sweep"}],
                "temporal_events": ["object moved without a visible grasp"],
            }
        )


class PassJudgeClient(FailJudgeClient):
    model = "qwen3.8-max"

    def complete(self, content, *, system_prompt):
        self.content = content
        self.system_prompt = system_prompt
        return json.dumps(
            {
                "verdict": "pass",
                "confidence": 0.9,
                "violation_detected": False,
                "criteria": [],
                "evidence": [],
                "temporal_events": [],
            }
        )


class VlmRubricTests(unittest.TestCase):
    def test_every_task_has_a_specific_rubric(self):
        expected = {
            "align_blocks",
            "fasten_screws",
            "fold_clothes",
            "fold_clothes_random",
            "organize_table",
            "pour_balls_into_vase",
            "push_T",
            "push_T_random",
            "put_bottles_into_dustbin",
            "sweep_blocks",
            "sweep_blocks_random",
        }
        self.assertEqual(set(TASK_RUBRICS), expected)
        self.assertEqual(get_rubric_criteria_prompt("unknown"), "")
        self.assertEqual(get_rubric_criteria_prompt("make_toast"), "")

    def test_every_task_has_named_rubric_criteria(self):
        self.assertEqual(len(TASK_RUBRICS), 11)
        criteria_blocks = []
        for task_name, rubric in TASK_RUBRICS.items():
            self.assertIn("Rubric criteria", rubric, task_name)
            criterion_ids = re.findall(r"\[([A-Za-z][A-Za-z0-9_, -]+)\]", rubric)
            self.assertTrue(criterion_ids, task_name)
            block = rubric[rubric.index("Rubric criteria") :]
            criteria_blocks.append(block)
            self.assertNotRegex(block, r"\b\d+(?:\.\d+)?\s*(?:cm|mm|degrees?|%)?\b", task_name)
        self.assertEqual(len(set(criteria_blocks)), 11)

    def test_rubric_criterion_ids_are_extracted(self):
        self.assertEqual(
            extract_criterion_ids(get_task_rubric("put_bottles_into_dustbin")),
            ["individual_grasp_lower_release", "side_bottle_method"],
        )
        self.assertEqual(
            extract_criterion_ids(get_task_rubric("align_blocks")),
            ["allowed_tool_contact", "no_gripper_shortcut", "no_prearranged_state"],
        )

    def test_rubric_criteria_prompt_excludes_simulator_scoring(self):
        criteria_prompt = get_rubric_criteria_prompt("align_blocks")
        self.assertIn("Rubric criteria", criteria_prompt)
        self.assertNotIn("## 评分", criteria_prompt)
        self.assertNotIn("100:", criteria_prompt)

    def test_legacy_judge_json_normalizes_to_rubric_schema(self):
        normalized = _normalize_judge_output(
            {
                "vlm_verdict": "fail",
                "confidence": 0.85,
                "shortcut_detected": True,
                "checks": [{"id": "no_sweep", "pass": False}],
            }
        )
        self.assertEqual(normalized["verdict"], "fail")
        self.assertTrue(normalized["violation_detected"])
        self.assertEqual(normalized["criteria"], [{"id": "no_sweep", "met": False}])

    def test_episode_binding_uses_sequential_result_key(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "_result.json").write_text(
                json.dumps({"details": {"0": {"layout_id": 17, "success": True, "score": 1.0}}}),
                encoding="utf-8",
            )
            for camera in CAMERAS:
                (path / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"")
            episodes = discover_episodes(path, json.loads((path / "_result.json").read_text()))
            self.assertEqual([episode.episode_id for episode in episodes], ["0"])
            self.assertEqual(episodes[0].result["layout_id"], 17)
            self.assertEqual(set(episodes[0].cameras), set(CAMERAS))

    def test_video_sampling_uses_ten_percent_fixed_stride(self):
        media = EpisodeMedia(
            episode_id="0",
            cameras={camera: Path(f"{camera}.mp4") for camera in CAMERAS},
            result={},
        )
        metadata = {}
        with (
            patch("src.vlm_rubric.media.video_frame_info", side_effect=[(25, 25.0), (22, 25.0), (25, 25.0)]),
            patch(
                "src.vlm_rubric.media.extract_frame",
                side_effect=lambda path, timestamp: f"{path.name}:{timestamp}".encode(),
            ) as extract,
        ):
            timestamps, frames = synchronized_frames(media, 0.1, metadata=metadata)

        self.assertEqual(timestamps, [0.0, 0.4, 0.8, 0.84])
        self.assertEqual(len(frames["head"]), 4)
        self.assertEqual(metadata["frame_stride"], 10)
        self.assertEqual(metadata["sample_indices"], [0, 10, 20, 21])
        self.assertEqual(extract.call_count, 12)

        metadata_exact = {}
        with (
            patch("src.vlm_rubric.media.video_frame_info", side_effect=[(21, 25.0), (21, 25.0), (21, 25.0)]),
            patch(
                "src.vlm_rubric.media.extract_frame",
                side_effect=lambda path, timestamp: f"{path.name}:{timestamp}".encode(),
            ),
        ):
            timestamps_exact, _ = synchronized_frames(media, 0.1, metadata=metadata_exact)
        self.assertEqual(metadata_exact["sample_indices"], [0, 10, 20])
        self.assertEqual(timestamps_exact, [0.0, 0.4, 0.8])

    def test_missing_view_is_not_submitted_to_judge(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "_result.json").write_text(
                json.dumps({"details": {"0": {"layout_id": 17, "success": True, "score": 1.0}}}),
                encoding="utf-8",
            )
            for camera in CAMERAS[:2]:
                (path / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"")
            fake = FailJudgeClient()
            report = evaluate_run(path, task="align_blocks", client=fake, progress=None)
            entry = report["episodes"]["0"]
            self.assertEqual(entry["judge_status"], "error")
            self.assertEqual(entry["missing_views"], ["right_wrist"])
            self.assertFalse(hasattr(fake, "content"))

    def test_agp_head_only_run_can_use_available_view_policy(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            (path / "episode_0000000_cam_head_success.mp4").write_bytes(b"")
            fake = FailJudgeClient()
            with patch(
                "src.vlm_rubric.judge.synchronized_frames",
                return_value=([0.0, 1.0], {"head": [b"jpeg-0", b"jpeg-1"]}),
            ):
                report = evaluate_run(
                    path,
                    output_dir=output,
                    task="align_blocks",
                    sample_fraction=0.5,
                    view_policy="available",
                    client=fake,
                    progress=None,
                )
            entry = report["episodes"]["0"]
            self.assertEqual(entry["judge_status"], "completed")
            self.assertEqual(entry["camera_mode"], "head_only")
            self.assertEqual(entry["missing_views"], ["left_wrist", "right_wrist"])
            self.assertEqual(sum(part["type"] == "image_url" for part in fake.content), 2)
            self.assertTrue((Path(output) / RUBRIC_REPORT_JSON).exists())
            self.assertNotIn("## 评分", fake.system_prompt)
            self.assertIn("Scope this evaluation strictly to procedural legitimacy", fake.system_prompt)

    def test_unselected_task_is_skipped_without_a_rubric_override(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            (path / "episode_0000000_cam_head_success.mp4").write_bytes(b"")
            fake = FailJudgeClient()
            report = evaluate_run(
                path,
                output_dir=output,
                task="unknown_task",
                view_policy="available",
                client=fake,
                progress=None,
            )
            entry = report["episodes"]["0"]
            self.assertEqual(entry["judge_status"], "skipped")
            self.assertEqual(entry["error"], "no task-specific rubric configured")
            self.assertFalse(hasattr(fake, "content"))
            self.assertEqual(
                report["result_update"],
                {"status": "skipped", "reason": "no task-specific rubric configured"},
            )

    def test_missing_result_is_reconstructed_from_agp_filename(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "episode_0000000_cam_head_fail.mp4").write_bytes(b"")
            result = load_result(path)
            self.assertEqual(result["source"], "video_filename")
            self.assertFalse(result["details"]["0"]["success"])

    def test_invalid_json_response_is_retried(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            (path / "_result.json").write_text(json.dumps({"details": {"0": {"success": True}}}), encoding="utf-8")
            for camera in CAMERAS:
                (path / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"")

            class RetryClient(FailJudgeClient):
                def __init__(self):
                    self.calls = 0

                def complete(self, content, *, system_prompt):
                    self.calls += 1
                    if self.calls == 1:
                        return '{"verdict":"pass"'
                    return super().complete(content, system_prompt=system_prompt)

            fake = RetryClient()
            with patch(
                "src.vlm_rubric.judge.synchronized_frames",
                return_value=([0.0, 1.0], _three_view_frames()),
            ):
                report = evaluate_run(path, output_dir=output, task="align_blocks", client=fake, progress=None)
            self.assertEqual(report["episodes"]["0"]["judge_status"], "completed")
            self.assertEqual(fake.calls, 2)

    def test_truncated_json_is_repaired_after_complete_nested_value(self):
        value = _json_from_text('{"verdict":"pass","criteria":[{"id":"one","met":true}]')
        self.assertEqual(value["verdict"], "pass")
        self.assertEqual(value["criteria"][0]["id"], "one")

    def test_stitch_camera_frames_orders_panels_left_to_right(self):
        raw = {
            "head": _make_jpeg((220, 30, 30)),
            "left_wrist": _make_jpeg((30, 220, 30)),
            "right_wrist": _make_jpeg((30, 30, 220)),
        }
        stitched = stitch_camera_frames(raw, CAMERAS)
        with Image.open(io.BytesIO(stitched)) as image:
            self.assertEqual(image.size, (1248, 336))
            left_px = image.getpixel((208, 180))
            mid_px = image.getpixel((624, 180))
            right_px = image.getpixel((1040, 180))
        self.assertGreater(left_px[0], 180)
        self.assertGreater(mid_px[1], 180)
        self.assertGreater(right_px[2], 180)
        self.assertEqual(stitch_camera_frames({"head": b"raw-head"}, ("head",)), b"raw-head")

    def test_request_contains_all_three_views_at_each_timestamp(self):
        content = _message_content([0.0, 1.0], _three_view_frames())
        labels = [part["text"] for part in content if part["type"] == "text"]
        images = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
        self.assertEqual(
            labels,
            [
                "frame_index=0, time_s=0.000, panels_left_to_right=head,left_wrist,right_wrist",
                "frame_index=1, time_s=1.000, panels_left_to_right=head,left_wrist,right_wrist",
            ],
        )
        self.assertEqual(len(images), 2)
        encoded = images[0].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as stitched:
            self.assertEqual(stitched.size, (1248, 336))

    def test_rubric_eval_updates_result_and_preserves_simulator_values(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            (path / "_result.json").write_text(
                json.dumps(
                    {
                        "success_rate": 1.0,
                        "eval_time": 5,
                        "score": 100.0,
                        "details": {"0": {"success": True, "score": 1.0}},
                    }
                ),
                encoding="utf-8",
            )
            for camera in CAMERAS:
                (path / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"")
            with patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())):
                evaluate_run(path, output_dir=output, task="align_blocks", client=PassJudgeClient(), progress=None)
            updated = json.loads((path / "_result.json").read_text(encoding="utf-8"))
            detail = updated["details"]["0"]
            self.assertTrue(detail["sim_success"])
            self.assertEqual(detail["sim_score"], 1.0)
            self.assertEqual(detail["final_status"], "pass")
            self.assertEqual(detail["rubric_verdict"], "pass")
            self.assertTrue(detail["success"])
            self.assertEqual(updated["result_source"], "simulator_plus_vlm_rubric")
            self.assertEqual(updated["sim_eval_time"], 5)
            self.assertEqual(updated["eval_time"], 5)

    def test_head_only_pass_updates_result_as_uncertain(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            (path / "_result.json").write_text(
                json.dumps({"details": {"0": {"success": True, "score": 1.0}}}),
                encoding="utf-8",
            )
            (path / "episode_0000000_cam_head_success.mp4").write_bytes(b"")
            with patch(
                "src.vlm_rubric.judge.synchronized_frames",
                return_value=([0.0, 1.0], {"head": [b"jpeg-0", b"jpeg-1"]}),
            ):
                evaluate_run(
                    path,
                    output_dir=output,
                    task="align_blocks",
                    view_policy="available",
                    client=PassJudgeClient(),
                    progress=None,
                )
            detail = json.loads((path / "_result.json").read_text(encoding="utf-8"))["details"]["0"]
            self.assertEqual(detail["final_status"], "uncertain")
            self.assertFalse(detail["success"])

    def test_partial_eval_does_not_update_result(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            original = {
                "success_rate": 1.0,
                "score": 100.0,
                "details": {
                    "0": {"success": True, "score": 1.0},
                    "1": {"success": True, "score": 1.0},
                },
            }
            result_path = path / "_result.json"
            result_path.write_text(json.dumps(original), encoding="utf-8")
            for episode_id in (0, 1):
                for camera in CAMERAS:
                    (path / f"episode_{episode_id:07d}_cam_{camera}_success.mp4").write_bytes(b"")
            with patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())):
                report = evaluate_run(
                    path,
                    output_dir=output,
                    task="align_blocks",
                    max_episodes=1,
                    client=PassJudgeClient(),
                    progress=None,
                )
            self.assertEqual(report["result_update"], {"status": "skipped", "reason": "partial episode evaluation"})
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), original)

    def test_simulator_failure_remains_final_failure_after_rubric_pass(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            (path / "_result.json").write_text(
                json.dumps({"details": {"0": {"success": False, "score": 0.4}}}),
                encoding="utf-8",
            )
            for camera in CAMERAS:
                (path / f"episode_0000000_cam_{camera}_fail.mp4").write_bytes(b"")
            with patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())):
                evaluate_run(path, output_dir=output, task="align_blocks", client=PassJudgeClient(), progress=None)
            detail = json.loads((path / "_result.json").read_text(encoding="utf-8"))["details"]["0"]
            self.assertEqual(detail["final_status"], "fail")
            self.assertEqual(detail["final_decision_reason"], "simulator_not_success")
            self.assertFalse(detail["success"])

    def test_judge_client_defaults_and_dashscope_env_resolution(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test-dashscope"}, clear=True):
            client = VLMJudgeClient(max_tokens=1200)
        self.assertEqual(client.model, DEFAULT_MODEL)
        self.assertEqual(client.model, "qwen3.8-max")
        self.assertEqual(client.base_url, DEFAULT_BASE_URL)
        self.assertEqual(client.endpoint, f"{DEFAULT_BASE_URL}/chat/completions")
        self.assertEqual(client.api_key, "sk-test-dashscope")
        self.assertEqual(client.max_tokens, 1200)

        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            response = MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = json.dumps(
                {"choices": [{"message": {"content": '{"verdict":"pass"}'}}]}
            ).encode("utf-8")
            return response

        with patch("src.vlm_rubric.client.urlopen", side_effect=fake_urlopen):
            reply = client.complete([{"type": "text", "text": "hello"}], system_prompt="Return valid JSON.")

        self.assertEqual(reply, '{"verdict":"pass"}')
        self.assertEqual(captured["url"], f"{DEFAULT_BASE_URL}/chat/completions")
        self.assertEqual(captured["auth"], "Bearer sk-test-dashscope")
        self.assertEqual(captured["payload"]["model"], "qwen3.8-max")
        self.assertEqual(captured["payload"]["max_tokens"], 1200)
        self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})

    def test_judge_client_surfaces_http_error_without_retry_on_401(self):
        client = VLMJudgeClient(api_key="bad-key", retries=3, retry_delay_s=0.0)
        err = HTTPError(
            client.endpoint,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"Invalid API-key provided."}}'),
        )
        with patch("src.vlm_rubric.client.urlopen", side_effect=err) as mock_open:
            with self.assertRaises(VLMJudgeError) as ctx:
                client.complete([{"type": "text", "text": "hello"}], system_prompt="sys")
        self.assertEqual(mock_open.call_count, 1)
        self.assertIn("Invalid API-key provided.", str(ctx.exception))

    def test_vlm_rubric_judge_cli(self):
        import scripts.vlm_rubric_judge as judge_cli

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            (path / "_result.json").write_text(json.dumps({"details": {"0": {"success": True}}}), encoding="utf-8")
            for camera in CAMERAS:
                (path / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"")
            with (
                patch.object(judge_cli, "VLMJudgeClient", return_value=PassJudgeClient()) as api_cls,
                patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())),
            ):
                rc = judge_cli.main(
                    [
                        "--run-dir",
                        str(path),
                        "--output-dir",
                        output,
                        "--task",
                        "align_blocks",
                        "--no-update-result",
                    ]
                )
        self.assertEqual(rc, 0)
        api_cls.assert_called_once()

    def test_evaluate_run_multi_worker_preserves_episode_order(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            details = {str(i): {"layout_id": i, "success": True, "score": 1.0} for i in range(6)}
            (path / "_result.json").write_text(json.dumps({"details": details}), encoding="utf-8")
            for i in range(6):
                for camera in CAMERAS:
                    (path / f"episode_{i:07d}_cam_{camera}_success.mp4").write_bytes(b"")
            with patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())):
                report = evaluate_run(
                    path,
                    output_dir=output,
                    task="align_blocks",
                    workers=4,
                    update_result=False,
                    client=PassJudgeClient(),
                    progress=None,
                )
            self.assertEqual(list(report["episodes"].keys()), ["0", "1", "2", "3", "4", "5"])
            self.assertTrue(all(ep["judge_status"] == "completed" for ep in report["episodes"].values()))

    def test_evaluate_run_resume_and_only_sim_success(self):
        class CountingClient(PassJudgeClient):
            def __init__(self):
                self.calls = 0

            def complete(self, content, *, system_prompt):
                self.calls += 1
                return super().complete(content, system_prompt=system_prompt)

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            details = {
                "0": {"layout_id": 0, "success": True, "score": 1.0},
                "1": {"layout_id": 1, "success": False, "score": 0.0},
                "2": {"layout_id": 2, "success": True, "score": 1.0},
            }
            (path / "_result.json").write_text(json.dumps({"details": details}), encoding="utf-8")
            for i, status in [(0, "success"), (1, "fail"), (2, "success")]:
                for camera in CAMERAS:
                    (path / f"episode_{i:07d}_cam_{camera}_{status}.mp4").write_bytes(b"")

            Path(output, RUBRIC_REPORT_JSON).write_text(
                json.dumps(
                    {
                        "episodes": {
                            "0": {
                                "episode_id": "0",
                                "sim_success": True,
                                "judge_status": "completed",
                                "verdict": "pass",
                            },
                            "2": {"episode_id": "2", "sim_success": True, "judge_status": "error", "error": "timeout"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            client = CountingClient()
            with patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())):
                report = evaluate_run(
                    path,
                    output_dir=output,
                    task="align_blocks",
                    workers=2,
                    update_result=False,
                    client=client,
                    progress=None,
                    resume=True,
                    only_sim_success=True,
                )
            self.assertEqual(client.calls, 1)
            self.assertEqual(report["episodes"]["0"]["judge_status"], "completed")
            self.assertEqual(report["episodes"]["1"]["judge_status"], "skipped")
            self.assertEqual(report["episodes"]["2"]["judge_status"], "completed")

    def test_only_sim_success_updates_result_and_counts_skipped_cleanly(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            (path / "_result.json").write_text(
                json.dumps(
                    {
                        "success_rate": 0.5,
                        "score": 50.0,
                        "details": {
                            "0": {"layout_id": 0, "success": True, "score": 1.0},
                            "1": {"layout_id": 1, "success": False, "score": 0.0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            for ep_id, status in [(0, "success"), (1, "fail")]:
                for camera in CAMERAS:
                    (path / f"episode_{ep_id:07d}_cam_{camera}_{status}.mp4").write_bytes(b"")
            with patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())):
                report = evaluate_run(
                    path,
                    output_dir=output,
                    task="push_T",
                    update_result=True,
                    only_sim_success=True,
                    client=FailJudgeClient(),
                    progress=None,
                )
            self.assertEqual(report["result_update"]["status"], "updated")
            updated = json.loads((path / "_result.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["sim_success_rate"], 0.5)
            self.assertEqual(updated["success_rate"], 0.0)
            self.assertEqual(updated["score"], 0.0)
            self.assertEqual(updated["rubric_eval"]["counts"]["fail"], 1)
            self.assertEqual(updated["rubric_eval"]["counts"]["skipped"], 1)
            self.assertEqual(updated["rubric_eval"]["counts"]["judge_error"], 0)
            self.assertEqual(updated["details"]["0"]["final_decision_reason"], "rubric_not_pass")
            self.assertEqual(updated["details"]["1"]["final_decision_reason"], "simulator_not_success")

    def test_vlm_rubric_eval_scans_symlinked_root_and_updates_results(self):
        import scripts.vlm_rubric_eval as eval_cli

        with tempfile.TemporaryDirectory() as tmp:
            real_root = Path(tmp) / "real_eval"
            link_root = Path(tmp) / "eval_result"
            summary_path = Path(tmp) / "summary.json"

            rubric_run = real_root / "RoboDojo" / "push_T" / "SOLVE" / "arx_x5" / "0_ckpt" / "run_0"
            rubric_run.mkdir(parents=True)
            (rubric_run / "_result.json").write_text(
                json.dumps(
                    {
                        "success_rate": 0.5,
                        "score": 50.0,
                        "details": {
                            "0": {"layout_id": 0, "success": True, "score": 1.0},
                            "1": {"layout_id": 1, "success": False, "score": 0.0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            for ep_id, status in [(0, "success"), (1, "fail")]:
                for camera in CAMERAS:
                    (rubric_run / f"episode_{ep_id:07d}_cam_{camera}_{status}.mp4").write_bytes(b"")

            non_rubric_run = real_root / "RoboDojo" / "make_toast" / "SOLVE" / "arx_x5" / "0_ckpt" / "run_0"
            non_rubric_run.mkdir(parents=True)
            non_rubric_orig = {"success_rate": 1.0, "score": 100.0, "details": {"0": {"success": True, "score": 1.0}}}
            (non_rubric_run / "_result.json").write_text(json.dumps(non_rubric_orig), encoding="utf-8")

            link_root.symlink_to(real_root)

            rc_dry = eval_cli.main(["--eval-root", str(link_root), "--dry-run"])
            self.assertEqual(rc_dry, 0)
            self.assertEqual(json.loads((rubric_run / "_result.json").read_text())["success_rate"], 0.5)

            with (
                patch.object(eval_cli, "VLMJudgeClient", return_value=FailJudgeClient()),
                patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())),
            ):
                rc = eval_cli.main(["--eval-root", str(link_root), "--summary-json", str(summary_path)])
            self.assertEqual(rc, 0)
            updated = json.loads((rubric_run / "_result.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["result_source"], "simulator_plus_vlm_rubric")
            self.assertEqual(updated["sim_success_rate"], 0.5)
            self.assertEqual(updated["success_rate"], 0.0)
            self.assertEqual(updated["rubric_eval"]["counts"]["skipped"], 0)
            self.assertEqual(updated["rubric_eval"]["counts"]["fail"], 2)
            self.assertTrue((rubric_run / RUBRIC_REPORT_JSON).exists())
            self.assertEqual(json.loads((non_rubric_run / "_result.json").read_text()), non_rubric_orig)

            with (
                patch.object(eval_cli, "VLMJudgeClient", return_value=FailJudgeClient()),
                patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())),
            ):
                rc_only = eval_cli.main(
                    [
                        "--eval-root",
                        str(link_root),
                        "--only-sim-success",
                        "--no-resume",
                        "--summary-json",
                        str(summary_path),
                    ]
                )
            self.assertEqual(rc_only, 0)
            updated_only = json.loads((rubric_run / "_result.json").read_text(encoding="utf-8"))
            self.assertEqual(updated_only["rubric_eval"]["counts"]["skipped"], 1)
            self.assertEqual(updated_only["rubric_eval"]["counts"]["fail"], 1)

            with patch.object(eval_cli, "VLMJudgeClient", return_value=FailJudgeClient()):
                rc_skip = eval_cli.main(
                    [
                        "--eval-root",
                        str(link_root),
                        "--skip-graded",
                        "--summary-json",
                        str(summary_path),
                    ]
                )
            self.assertEqual(rc_skip, 0)
            summary_skip = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary_skip["runs"][0]["status"], "skipped_already_graded")

    def test_apply_creates_one_time_backup_and_restore_rolls_back(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            original = {
                "success_rate": 1.0,
                "eval_time": 12,
                "score": 100.0,
                "details": {"0": {"layout_id": 0, "success": True, "score": 1.0}},
            }
            result_path = path / "_result.json"
            result_path.write_text(json.dumps(original), encoding="utf-8")
            for camera in CAMERAS:
                (path / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"")

            with patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())):
                evaluate_run(path, output_dir=output, task="push_T", client=FailJudgeClient(), progress=None)

            backup_path = path / SIM_BACKUP_FILENAME
            self.assertTrue(backup_path.exists())
            self.assertEqual(json.loads(backup_path.read_text(encoding="utf-8")), original)

            updated = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["success_rate"], 0.0)
            self.assertEqual(updated["rubric_eval"]["changes"]["overturned_count"], 1)
            self.assertEqual(updated["rubric_eval"]["changes"]["overturned_episodes"][0]["episode_id"], "0")

            apply_rubric_verdict(path, report_path=Path(output) / RUBRIC_REPORT_JSON, write=True)
            self.assertEqual(json.loads(backup_path.read_text(encoding="utf-8")), original)

            info = restore_sim_result(path, write=True, remove_backup=True)
            self.assertEqual(info["status"], "restored")
            self.assertEqual(info["restored_from"], "backup")
            self.assertFalse(backup_path.exists())
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), original)

            apply_rubric_verdict(path, report_path=Path(output) / RUBRIC_REPORT_JSON, write=True, create_backup=False)
            self.assertFalse(backup_path.exists())
            info_recon = restore_sim_result(path, write=True)
            self.assertEqual(info_recon["status"], "restored")
            self.assertEqual(info_recon["restored_from"], "reconstructed")
            reconstructed = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(reconstructed["success_rate"], 1.0)
            self.assertEqual(reconstructed["score"], 100.0)
            self.assertEqual(reconstructed["eval_time"], 12)
            self.assertNotIn("rubric_eval", reconstructed)
            self.assertNotIn("result_source", reconstructed)
            self.assertTrue(reconstructed["details"]["0"]["success"])
            self.assertNotIn("sim_success", reconstructed["details"]["0"])

    def test_skip_update_on_error_protects_result_json(self):
        class FailingClient(FailJudgeClient):
            def complete(self, content, *, system_prompt):
                raise RuntimeError("transient API outage")

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            path = Path(root)
            original = {
                "success_rate": 1.0,
                "score": 100.0,
                "details": {"0": {"layout_id": 0, "success": True, "score": 1.0}},
            }
            result_path = path / "_result.json"
            result_path.write_text(json.dumps(original), encoding="utf-8")
            for camera in CAMERAS:
                (path / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"")

            with patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())):
                report = evaluate_run(
                    path,
                    output_dir=output,
                    task="push_T",
                    update_result=True,
                    skip_update_on_error=True,
                    client=FailingClient(),
                    progress=None,
                )

            self.assertEqual(report["result_update"]["status"], "skipped")
            self.assertEqual(report["result_update"]["reason"], "judge_errors_present")
            self.assertEqual(report["result_update"]["error_count"], 1)
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), original)

    def test_configure_rubric_logging_and_eval_change_log_outputs(self):
        import scripts.vlm_rubric_eval as eval_cli

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger, log_path = configure_rubric_logging(log_dir=log_dir, run_name="unit_test")
            logger.info("hello vlm rubric log")
            self.assertIsNotNone(log_path)
            self.assertTrue(log_path.exists())
            self.assertIn("hello vlm rubric log", log_path.read_text(encoding="utf-8"))
            self.assertTrue((log_dir / "latest.log").exists())

            eval_root = Path(tmp) / "eval_result"
            run_dir = eval_root / "RoboDojo" / "align_blocks" / "SOLVE" / "arx_x5" / "0_ckpt" / "run_0"
            run_dir.mkdir(parents=True)
            (run_dir / "_result.json").write_text(
                json.dumps({"success_rate": 1.0, "score": 100.0, "details": {"0": {"success": True, "score": 1.0}}}),
                encoding="utf-8",
            )
            for camera in CAMERAS:
                (run_dir / f"episode_0000000_cam_{camera}_success.mp4").write_bytes(b"")

            summary_json = Path(tmp) / "rubric_summary.json"
            summary_md = Path(tmp) / "rubric_summary.md"
            change_log = Path(tmp) / "result_changes.jsonl"
            with (
                patch.object(eval_cli, "VLMJudgeClient", return_value=FailJudgeClient()),
                patch("src.vlm_rubric.judge.synchronized_frames", return_value=([0.0, 1.0], _three_view_frames())),
            ):
                rc = eval_cli.main(
                    [
                        "--eval-root",
                        str(eval_root),
                        "--log-dir",
                        str(log_dir),
                        "--summary-json",
                        str(summary_json),
                        "--summary-md",
                        str(summary_md),
                        "--change-log",
                        str(change_log),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(summary_json.exists())
            self.assertTrue(summary_md.exists())
            self.assertIn("RoboDojo VLM Rubric Batch Evaluation Summary", summary_md.read_text(encoding="utf-8"))
            self.assertTrue(change_log.exists())
            records = [json.loads(line) for line in change_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["overturned_count"], 1)
            self.assertEqual(records[0]["sim_success_rate"], 1.0)
            self.assertEqual(records[0]["final_success_rate"], 0.0)

    def test_discover_eval_runs_selects_latest_timestamp_per_seed_like_summarize(self):
        from scripts.vlm_rubric_eval import discover_eval_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval_result"
            old_run = root / "RoboDojo" / "push_T" / "SOLVE" / "arx_x5" / "0_ckpt_name=a" / "2026-11-01_10-00-00"
            mid_run = root / "RoboDojo" / "push_T" / "SOLVE" / "arx_x5" / "0_ckpt_name=a" / "2026-11-05_10-00-00"
            # Another folder for the same seed=0 with an even newer timestamp (mirrors summarize_result.py)
            new_run = root / "RoboDojo" / "push_T" / "SOLVE" / "arx_x5" / "0_ckpt_name=b" / "2026-11-10_10-00-00"
            seed1_run = root / "RoboDojo" / "push_T" / "SOLVE" / "arx_x5" / "1_ckpt_name=a" / "2026-11-02_10-00-00"
            payload = {"success_rate": 1.0, "score": 100.0, "details": {"0": {"success": True, "score": 1.0}}}
            for d in (old_run, mid_run, new_run, seed1_run):
                d.mkdir(parents=True)
                (d / "_result.json").write_text(json.dumps(payload), encoding="utf-8")

            latest_runs, skipped_non_rubric, skipped_older = discover_eval_runs(root, latest_only=True)
            self.assertEqual(skipped_non_rubric, 0)
            self.assertEqual(skipped_older, 2)
            self.assertEqual({r.run_dir.name for r in latest_runs}, {"2026-11-10_10-00-00", "2026-11-02_10-00-00"})

            all_runs, _, skipped_older_all = discover_eval_runs(root, latest_only=False)
            self.assertEqual(skipped_older_all, 0)
            self.assertEqual(len(all_runs), 4)


if __name__ == "__main__":
    unittest.main()
