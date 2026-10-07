import copy
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("h3lv_test")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("h3lv_test", package)
controller = importlib.import_module("h3lv_test.controller")
core = importlib.import_module("h3lv_test.core")


def native_prompt():
    return {
        "1": {"class_type": "H3LVUnified", "inputs": {}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 2]}},
        "3": {"class_type": "CreateVideo", "inputs": {
            "images": ["2", 0], "audio": ["1", 0], "fps": 30}},
        "4": {"class_type": "SaveVideo", "inputs": {
            "video": ["3", 0], "filename_prefix": "video/ComfyUI",
            "format": "auto", "codec": "auto"}},
    }


class VideoOutputTests(unittest.TestCase):
    def test_native_history_is_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)/"project"
            project.mkdir()
            for extension in ("mp4", "webm", "mkv"):
                with self.subTest(extension=extension):
                    video = project/f"native.{extension}"
                    video.write_bytes(b"test")
                    history = {"status": {"status_str": "success"}, "outputs": {
                        "4": {"images": [{"filename": video.name, "subfolder": "project",
                                            "type": "output"}], "animated": [True]}}}
                    self.assertEqual(controller.video_from_history(history, "4", project, directory), str(video))
                    preview = core.output_preview(directory, video)
                    self.assertEqual(preview["format"],
                        "video/h264-mp4" if extension == "mp4" else f"video/{extension}")

    def test_native_output_binding_does_not_change_canvas_or_encoding_settings(self):
        prompt = native_prompt()
        original = copy.deepcopy(prompt)
        submitted = copy.deepcopy(prompt)
        controller.bind_video_output(submitted, "1", "4")
        self.assertEqual(submitted["4"]["inputs"]["filename_prefix"], ["1", 3])
        self.assertEqual(submitted["3"]["inputs"]["fps"], ["1", 12])
        self.assertEqual(submitted["4"]["inputs"]["codec"], "auto")
        self.assertEqual(prompt, original)

    def test_native_history_retains_project_path_and_success_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)/"project"
            project.mkdir()
            (Path(directory)/"outside.mp4").write_bytes(b"test")
            history = {"status": {"status_str": "success"}, "outputs": {
                "4": {"images": [{"filename": "outside.mp4", "type": "output"}]}}}
            with self.assertRaises(ValueError):
                controller.video_from_history(history, "4", project, directory)
            history["status"]["status_str"] = "error"
            with self.assertRaisesRegex(RuntimeError, "生成失败"):
                controller.video_from_history(history, "4", project, directory)

    def test_vhs_binding_preserves_encoding_and_other_savers(self):
        prompt = native_prompt()
        prompt["5"] = {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["2", 0], "frame_rate": 8, "save_output": False,
            "format": "video/h264-mp4", "crf": 19}}
        native = copy.deepcopy(prompt["4"])
        controller.bind_video_output(prompt, "1", "5")
        self.assertEqual(prompt["5"]["inputs"]["frame_rate"], ["1", 12])
        self.assertTrue(prompt["5"]["inputs"]["save_output"])
        self.assertEqual(prompt["5"]["inputs"]["crf"], 19)
        self.assertEqual(prompt["4"], native)

    def test_native_save_requires_create_video_and_keeps_current_snapshot_slots(self):
        prompt = native_prompt()
        snapshot = {"prompt": prompt, "loader_id": "1", "video_id": "4",
                    "output_contract_version": controller.OUTPUT_CONTRACT_VERSION}
        prompt["3"]["inputs"]["fps"] = ["1", 12]
        controller.normalize_output_contract(snapshot)
        self.assertEqual(prompt["3"]["inputs"]["fps"], ["1", 12])
        prompt["3"]["class_type"] = "LoadVideo"
        with self.assertRaisesRegex(ValueError, "CreateVideo"):
            controller.bind_video_output(prompt, "1", "4")

    def test_multiple_output_fingerprint_includes_selected_saver(self):
        snapshot = {"prompt": native_prompt(), "loader_id": "1", "video_id": "4"}
        snapshot["prompt"]["5"] = copy.deepcopy(snapshot["prompt"]["4"])
        first = controller.generation_graph_fingerprint(snapshot)
        snapshot["video_id"] = "5"
        self.assertNotEqual(controller.generation_graph_fingerprint(snapshot), first)

    def test_native_collection_ignores_images_temp_files_and_other_node_results(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)/"project"
            project.mkdir()
            for name in ("picture.png", "temporary.mp4", "other.mp4"):
                (project/name).write_bytes(b"test")
            history = {"status": {"status_str": "success"}, "outputs": {
                "4": {"images": [{"filename": "picture.png", "subfolder": "project", "type": "output"},
                                  {"filename": "temporary.mp4", "subfolder": "project", "type": "temp"}]},
                "5": {"images": [{"filename": "other.mp4", "subfolder": "project", "type": "output"}]}}}
            with self.assertRaisesRegex(RuntimeError, "所选输出节点"):
                controller.video_from_history(history, "4", project, directory)


if __name__ == "__main__":
    unittest.main()
