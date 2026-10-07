"""CPU-only integration: real ComfyUI executor, native/VHS saves and H3 assembly.

Run with ComfyUI's Python: native_video_smoke.py COMFYUI_ROOT
All generated audio/video is confined to an automatically cleaned temporary folder.
Synthetic frames stand in for diffusion; no model download or paid API is used.
"""
import asyncio
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(COMFY_ROOT))
from comfy.cli_args import args
args.cpu = True

import folder_paths
import nodes as comfy_nodes
import execution
import soundfile as sf
import torch
from comfy_extras.nodes_video import CreateVideo, SaveVideo

package = types.ModuleType("h3lv_smoke")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
core = importlib.import_module("h3lv_smoke.core")
controller = importlib.import_module("h3lv_smoke.controller")
plugin_nodes = importlib.import_module("h3lv_smoke.nodes")
rules = importlib.import_module("h3lv_smoke.director_rules")


class SmokeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "make"
    CATEGORY = "test"

    def make(self):
        waveform = torch.sin(torch.arange(144000).float() * (2 * torch.pi * 220 / 48000)) * .1
        return ({"waveform": waveform[None, None, :], "sample_rate": 48000},)


class SmokeFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("INT",), "audio": ("AUDIO",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make"
    CATEGORY = "test"

    def make(self, frames, audio):
        # Small deterministic moving colour ramp, with more frames than the edit needs.
        ramp = torch.linspace(0, 1, frames)[:, None, None, None]
        return (ramp.expand(frames, 64, 96, 3).contiguous(),)


comfy_nodes.NODE_CLASS_MAPPINGS.update({
    "H3LVUnified": plugin_nodes.Unified,
    "H3LVSmokeAudio": SmokeAudio, "H3LVSmokeFrames": SmokeFrames,
    "CreateVideo": CreateVideo, "SaveVideo": SaveVideo,
})


async def run_case(output_root, label, output_type, container="auto", extra_output=False):
    folder_paths.set_output_directory(str(output_root))
    root = plugin_nodes.data_root()
    plan = core.decorate({
        "id": uuid.uuid4().hex, "sample_rate": 48000, "samples": 144000, "duration": 3,
        "mode": "speaking", "max_seconds": 5, "target_seconds": 5,
        "created": 0, "revision": 1, "approved": False, "run_status": "draft",
        "director": {"performance_intensity": "auto", "note": "",
                     "rule_config": rules.default_config(), "schedule_seed": "smoke", "rule_revision": "test"},
        "segments": [{"start_sample": i*48000, "end_sample": (i+1)*48000,
                      "energy_db": -20, "text": "测试"} for i in range(3)],
    })
    plan["approved"] = True
    plan["approved_fingerprint"] = core.fingerprint(plan)
    project = core.project_path(root, plan["id"])
    project.mkdir(parents=True)
    core.audio_file(project, "source.wav").parent.mkdir()
    audio = SmokeAudio().make()[0]["waveform"].numpy()[0].T
    for name in ("source.wav", "vocals.wav"):
        sf.write(core.audio_file(project, name), audio, 48000)
    core.write_plan(root, plan)
    prompt = {
        "0": {"class_type": "H3LVSmokeAudio", "inputs": {}},
        "1": {"class_type": "H3LVUnified", "inputs": {
            "audio": ["0", 0], "mode": "speaking", "max_seconds": 5., "target_seconds": 5.,
            "asr_python": "", "asr_model": "", "asr_device": "cpu",
            "director_mode": "本地规则", "project_id": plan["id"], "segment_index": 0}},
        "2": {"class_type": "H3LVSmokeFrames", "inputs": {"frames": ["1", 2], "audio": ["1", 0]}},
    }
    if output_type == "SaveVideo":
        prompt["3"] = {"class_type": "CreateVideo", "inputs": {"images": ["2", 0], "audio": ["1", 0], "fps": 30.}}
        prompt["4"] = {"class_type": "SaveVideo", "inputs": {
            "video": ["3", 0], "filename_prefix": "video/default", "format": container}}
        if container == "webm":
            prompt["4"]["inputs"]["format.codec"] = "av1"
    else:
        prompt["4"] = {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["2", 0], "audio": ["1", 0], "frame_rate": 8,
            "filename_prefix": "video/default", "format": "video/h264-mp4",
            "loop_count": 0, "pingpong": False, "save_output": False,
            "pix_fmt": "yuv420p", "crf": 19, "save_metadata": False, "trim_to_audio": False}}
    if extra_output:
        prompt["5"] = {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["2", 0], "frame_rate": 24, "filename_prefix": "must_not_be_written",
            "format": "video/h264-mp4", "loop_count": 0, "pingpong": False, "save_output": True}}
    histories, running, events, indices = {}, [], [], []
    server = types.SimpleNamespace(number=0, client_id=None, last_node_id=None,
        send_sync=lambda name, data, *unused: events.append((name, data)))
    executor = execution.PromptExecutor(server, cache_type=execution.CacheType.NONE,
        cache_args={"ram": 0, "ram_inactive": 0})

    async def consume(item):
        try:
            indices.append(item[2]["1"]["inputs"]["segment_index"])
            await executor.execute_async(item[2], item[1], item[3], item[4])
            histories[item[1]] = dict(executor.history_result,
                status={"status_str": "success" if executor.success else "error"})
        except Exception as exc:
            histories[item[1]] = {"status": {"status_str": "error"}, "outputs": {}}
            print(f"Executor harness failed: {type(exc).__name__}: {exc}")
        finally:
            running.remove(item)

    class Queue:
        def put(self, item):
            running.append(item)
            asyncio.create_task(consume(item))
        def get_history(self, prompt_id):
            return {prompt_id: histories[prompt_id]} if prompt_id in histories else {}
        def get_current_queue(self):
            return running, []
    server.prompt_queue = Queue()
    controller.start(root, plan["id"], {"prompt": prompt, "loader_id": "1", "video_id": "4"}, server)
    await controller.TASKS[plan["id"]]
    saved = core.read_plan(root, plan["id"])
    assert saved["run_status"] == "completed", saved.get("error")
    assert indices == [0, 1, 2], indices
    assert saved["video_output_id"] == "4"
    for row in saved["segments"]:
        info = controller.probe_video(row["job"]["video"])
        assert info["r_frame_rate"] == "24/1", info
        assert int(info["nb_read_frames"]) >= row["edit_frames"], info
        suffix = {"auto": ".mp4", "mkv": ".mkv", "webm": ".webm"}[container]
        assert Path(row["job"]["video"]).suffix == suffix
    if extra_output:
        assert all("5" not in history["outputs"] for history in histories.values())
        assert not list(output_root.glob("must_not_be_written*"))
    final = controller.probe_video(saved["final_video"])
    assert final["r_frame_rate"] == "24/1" and int(final["nb_read_frames"]) == 72, final
    assert abs(float(final["duration"])-3) < .001, final
    tracks = json.loads(controller.command([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
        "-of", "json", saved["final_video"]]))["streams"]
    audio_track = next(track for track in tracks if track["codec_type"] == "audio")
    assert abs(float(audio_track["duration"])-3) < .001, tracks
    final_event = next(data for name, data in events if name == "h3lv-final")
    assert final_event["video_id"] == "4"
    if extra_output:
        # Exercise the separate "assemble only" route with both savers present.
        from aiohttp import web
        from server import PromptServer
        plugin_routes = importlib.import_module("h3lv_smoke.routes")
        server.routes = web.RouteTableDef()
        PromptServer.instance = server
        plugin_routes.register_routes()
        handler = next(route.handler for route in server.routes
                       if route.path == "/h3lv/project/{project_id}/assemble")
        response = await handler(types.SimpleNamespace(match_info={"project_id": plan["id"]}))
        assert response.status == 200, response.text
        reassembled = json.loads(response.text)
        assert reassembled["video_output_id"] == "4"
        assert reassembled["final_video"] != saved["final_video"]
        assert [data for name, data in events if name == "h3lv-final"][-1]["video_id"] == "4"
    print(json.dumps({"case": label, "segments": len(indices), "frames": int(final["nb_read_frames"]),
                      "fps": final["r_frame_rate"], "duration": final["duration"], "result": "PASS"}))


async def main():
    with tempfile.TemporaryDirectory(prefix="h3lv_video_output_smoke_") as directory:
        output_root = Path(directory)/"output"
        output_root.mkdir()
        for container in ("auto", "mkv", "webm"):
            await run_case(output_root, f"native-{container}", "SaveVideo", container)
        vhs_root = COMFY_ROOT/"custom_nodes"/"ComfyUI-VideoHelperSuite"
        if vhs_root.is_dir():
            sys.path.insert(0, str(vhs_root))
            from server import PromptServer
            PromptServer.instance = types.SimpleNamespace(prompt_queue=None)
            from videohelpersuite.nodes import VideoCombine
            comfy_nodes.NODE_CLASS_MAPPINGS["VHS_VideoCombine"] = VideoCombine
            await run_case(output_root, "legacy-vhs-mp4", "VHS_VideoCombine")
            await run_case(output_root, "mixed-native-selected", "SaveVideo", extra_output=True)
        else:
            print("VHS smoke skipped: VideoHelperSuite is not installed.")


if __name__ == "__main__":
    asyncio.run(main())
