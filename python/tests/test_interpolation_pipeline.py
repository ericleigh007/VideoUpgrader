import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from upscaler_worker.interpolation import resolve_segment_output_frame_count
from upscaler_worker.pipeline import PipelineSegment, _plan_interpolation_segments, run_realesrgan_pipeline


class InterpolationPipelineTests(unittest.TestCase):
    def test_plan_interpolation_segments_adds_boundary_overlap_and_trim_windows(self) -> None:
        segments = [
            PipelineSegment(index=0, start_frame=0, frame_count=240, start_seconds=0.0, duration_seconds=10.0),
            PipelineSegment(index=1, start_frame=240, frame_count=60, start_seconds=10.0, duration_seconds=2.5),
        ]

        plans = _plan_interpolation_segments(
            segments,
            total_source_frames=300,
            source_fps=24.0,
            output_fps=60.0,
        )

        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].expanded_frame_count, 241)
        self.assertEqual(plans[0].overlap_before_frames, 0)
        self.assertEqual(plans[0].overlap_after_frames, 1)
        self.assertEqual(plans[0].output_frame_count, 600)
        self.assertEqual(plans[1].expanded_start_frame, 239)
        self.assertEqual(plans[1].expanded_frame_count, 61)
        self.assertEqual(plans[1].overlap_before_frames, 1)
        self.assertEqual(plans[1].overlap_after_frames, 0)
        self.assertEqual(plans[1].output_start_frame, 2)
        self.assertEqual(plans[1].output_frame_count, 150)

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", side_effect=[(750, 750), (750, 750)])
    @patch("upscaler_worker.pipeline._concat_segment_videos", return_value=750)
    @patch("upscaler_worker.pipeline._encode_segment_video", side_effect=[600, 150])
    @patch("upscaler_worker.pipeline._run_rife_segment", side_effect=[603, 153])
    @patch("upscaler_worker.pipeline._upscale_ncnn_segment", side_effect=[241, 61])
    @patch("upscaler_worker.pipeline._extract_segment_frames", side_effect=[241, 61])
    @patch("upscaler_worker.pipeline.ensure_rife_runtime")
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_backend_id", return_value="realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    def test_after_upscale_interpolation_pipeline_reports_output_frame_counts(
        self,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        ensure_runtime_assets_mock,
        probe_video_mock,
        ensure_rife_runtime_mock,
        extract_mock,
        upscale_mock,
        rife_mock,
        encode_mock,
        concat_mock,
        ffmpeg_progress_mock,
    ) -> None:
        ensure_runtime_assets_mock.return_value = {
            "ffmpegPath": "ffmpeg",
            "realesrganPath": "realesrgan.exe",
            "modelDir": "models/realesrgan",
            "availableGpus": [],
            "defaultGpuId": None,
        }
        ensure_rife_runtime_mock.return_value = {
            "rifePath": "rife.exe",
            "rifeModelRoot": "models/rife",
        }
        probe_video_mock.return_value = {
            "width": 1280,
            "height": 720,
            "frameRate": 24.0,
            "durationSeconds": 12.5,
            "hasAudio": True,
            "videoCodec": "h264",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "output.mp4"
            progress_path = temp_root / "progress.json"

            result = run_realesrgan_pipeline(
                source_path="C:/fixtures/input.mp4",
                model_id="realesrgan-x4plus",
                output_mode="preserveAspect4k",
                preset="qualityBalanced",
                interpolation_mode="afterUpscale",
                interpolation_target_fps=60,
                gpu_id=0,
                aspect_ratio_preset="16:9",
                custom_aspect_width=None,
                custom_aspect_height=None,
                resolution_basis="exact",
                target_width=3840,
                target_height=2160,
                crop_left=None,
                crop_top=None,
                crop_width=None,
                crop_height=None,
                progress_path=str(progress_path),
                cancel_path=None,
                preview_mode=False,
                preview_duration_seconds=None,
                segment_duration_seconds=10.0,
                output_path=str(output_path),
                codec="h264",
                container="mp4",
                tile_size=0,
                fp16=False,
                torch_compile_enabled=False,
                crf=18,
            )

            self.assertEqual(result["frameCount"], 750)
            self.assertIn("interpolateSeconds", result["stageTimings"])
            self.assertEqual(result["segmentCount"], 2)
            self.assertEqual(result["executionPath"], "rife-ncnn-vulkan")
            self.assertIn("Interpolation mode: afterUpscale", "\n".join(result["log"]))
            self.assertIn("Chunked interpolation segments: 2", "\n".join(result["log"]))

            self.assertEqual(extract_mock.call_count, 2)
            self.assertEqual(upscale_mock.call_count, 2)
            self.assertEqual(rife_mock.call_count, 2)
            self.assertEqual(encode_mock.call_count, 2)
            concat_mock.assert_called_once()
            self.assertEqual(ffmpeg_progress_mock.call_count, 1)

            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["phase"], "completed")
            self.assertEqual(progress_payload["totalFrames"], 750)
            self.assertEqual(progress_payload["interpolatedFrames"], 750)
            self.assertEqual(progress_payload["remuxedFrames"], 750)

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", side_effect=[(745, 745), (745, 745)])
    @patch("upscaler_worker.pipeline._concat_segment_videos", return_value=745)
    @patch("upscaler_worker.pipeline._encode_segment_video", side_effect=[600, 145])
    @patch("upscaler_worker.pipeline._run_rife_segment", side_effect=[603, 147])
    @patch("upscaler_worker.pipeline._upscale_ncnn_segment", side_effect=[241, 59])
    @patch("upscaler_worker.pipeline._extract_segment_frames", side_effect=[241, 59])
    @patch("upscaler_worker.pipeline.ensure_rife_runtime")
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_backend_id", return_value="realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    def test_after_upscale_interpolation_tolerates_short_final_segment_at_eof(
        self,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        ensure_runtime_assets_mock,
        probe_video_mock,
        ensure_rife_runtime_mock,
        extract_mock,
        upscale_mock,
        rife_mock,
        encode_mock,
        concat_mock,
        ffmpeg_progress_mock,
    ) -> None:
        ensure_runtime_assets_mock.return_value = {
            "ffmpegPath": "ffmpeg",
            "realesrganPath": "realesrgan.exe",
            "modelDir": "models/realesrgan",
            "availableGpus": [],
            "defaultGpuId": None,
        }
        ensure_rife_runtime_mock.return_value = {
            "rifePath": "rife.exe",
            "rifeModelRoot": "models/rife",
        }
        probe_video_mock.return_value = {
            "width": 1280,
            "height": 720,
            "frameRate": 24.0,
            "durationSeconds": 12.5,
            "hasAudio": True,
            "videoCodec": "h264",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "output.mp4"
            progress_path = temp_root / "progress.json"

            result = run_realesrgan_pipeline(
                source_path="C:/fixtures/input.mp4",
                model_id="realesrgan-x4plus",
                output_mode="preserveAspect4k",
                preset="qualityBalanced",
                interpolation_mode="afterUpscale",
                interpolation_target_fps=60,
                gpu_id=0,
                aspect_ratio_preset="16:9",
                custom_aspect_width=None,
                custom_aspect_height=None,
                resolution_basis="exact",
                target_width=3840,
                target_height=2160,
                crop_left=None,
                crop_top=None,
                crop_width=None,
                crop_height=None,
                progress_path=str(progress_path),
                cancel_path=None,
                preview_mode=False,
                preview_duration_seconds=None,
                segment_duration_seconds=10.0,
                output_path=str(output_path),
                codec="h264",
                container="mp4",
                tile_size=0,
                fp16=False,
                torch_compile_enabled=False,
                crf=18,
            )

            expected_trimmed_output_frames = resolve_segment_output_frame_count(
                start_frame=240,
                frame_count=58,
                source_fps=24.0,
                output_fps=60.0,
            )

            self.assertEqual(result["frameCount"], 745)
            self.assertEqual(result["interpolationDiagnostics"]["sourceFrameCount"], 298)
            self.assertEqual(result["interpolationDiagnostics"]["outputFrameCount"], 745)
            self.assertEqual(encode_mock.call_args_list[1].kwargs["input_frame_limit"], expected_trimmed_output_frames)
            self.assertIn("reached EOF early", "\n".join(result["log"]))
            concat_mock.assert_called_once()
            self.assertEqual(ffmpeg_progress_mock.call_count, 1)

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", side_effect=[(750, 750), (750, 750)])
    @patch("upscaler_worker.pipeline._concat_segment_videos", return_value=750)
    @patch("upscaler_worker.pipeline._encode_segment_video")
    @patch("upscaler_worker.pipeline._run_rife_segment")
    @patch("upscaler_worker.pipeline._upscale_ncnn_segment")
    @patch("upscaler_worker.pipeline._extract_segment_frames")
    @patch("upscaler_worker.pipeline.ensure_rife_runtime")
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_backend_id", return_value="realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    def test_after_upscale_interpolation_overlaps_second_segment_with_first_encode(
        self,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        ensure_runtime_assets_mock,
        probe_video_mock,
        ensure_rife_runtime_mock,
        extract_mock,
        upscale_mock,
        rife_mock,
        encode_mock,
        concat_mock,
        ffmpeg_progress_mock,
    ) -> None:
        ensure_runtime_assets_mock.return_value = {
            "ffmpegPath": "ffmpeg",
            "realesrganPath": "realesrgan.exe",
            "modelDir": "models/realesrgan",
            "availableGpus": [],
            "defaultGpuId": None,
        }
        ensure_rife_runtime_mock.return_value = {
            "rifePath": "rife.exe",
            "rifeModelRoot": "models/rife",
        }
        probe_video_mock.return_value = {
            "width": 1280,
            "height": 720,
            "frameRate": 24.0,
            "durationSeconds": 12.5,
            "hasAudio": True,
            "videoCodec": "h264",
        }

        first_encode_started = threading.Event()
        second_segment_interpolation_started = threading.Event()
        overlap_verified = {"value": False}

        def extract_side_effect(*args, **kwargs):
            segment = kwargs["segment"]
            return segment.frame_count

        def upscale_side_effect(*args, **kwargs):
            output_dir = kwargs["output_dir"]
            return 241 if "segment_0000" in str(output_dir) else 61

        def rife_side_effect(*args, **kwargs):
            output_dir = kwargs["output_dir"]
            if "segment_0000" in str(output_dir):
                return 603
            overlap_verified["value"] = first_encode_started.is_set()
            second_segment_interpolation_started.set()
            return 153

        def encode_side_effect(*args, **kwargs):
            output_file = kwargs["output_file"]
            if "segment_0000" in str(output_file):
                first_encode_started.set()
                self.assertTrue(
                    second_segment_interpolation_started.wait(timeout=2.0),
                    "Second segment interpolation never started while first encode was active",
                )
                return 600
            return 150

        extract_mock.side_effect = extract_side_effect
        upscale_mock.side_effect = upscale_side_effect
        rife_mock.side_effect = rife_side_effect
        encode_mock.side_effect = encode_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "output.mp4"
            progress_path = temp_root / "progress.json"

            result = run_realesrgan_pipeline(
                source_path="C:/fixtures/input.mp4",
                model_id="realesrgan-x4plus",
                output_mode="preserveAspect4k",
                preset="qualityBalanced",
                interpolation_mode="afterUpscale",
                interpolation_target_fps=60,
                gpu_id=0,
                aspect_ratio_preset="16:9",
                custom_aspect_width=None,
                custom_aspect_height=None,
                resolution_basis="exact",
                target_width=3840,
                target_height=2160,
                crop_left=None,
                crop_top=None,
                crop_width=None,
                crop_height=None,
                progress_path=str(progress_path),
                cancel_path=None,
                preview_mode=False,
                preview_duration_seconds=None,
                segment_duration_seconds=10.0,
                output_path=str(output_path),
                codec="h264",
                container="mp4",
                tile_size=0,
                fp16=False,
                torch_compile_enabled=False,
                crf=18,
            )

            self.assertEqual(result["frameCount"], 750)
            self.assertTrue(overlap_verified["value"])
            self.assertEqual(extract_mock.call_count, 2)
            self.assertEqual(upscale_mock.call_count, 2)
            self.assertEqual(rife_mock.call_count, 2)
            self.assertEqual(encode_mock.call_count, 2)
            concat_mock.assert_called_once()
            self.assertEqual(ffmpeg_progress_mock.call_count, 1)

    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_rife_runtime")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_backend_id", return_value="realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    def test_interpolation_rejects_lower_target_fps(
        self,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        ensure_runtime_assets_mock,
        ensure_rife_runtime_mock,
        probe_video_mock,
    ) -> None:
        ensure_runtime_assets_mock.return_value = {
            "ffmpegPath": "ffmpeg",
            "realesrganPath": "realesrgan.exe",
            "modelDir": "models/realesrgan",
            "availableGpus": [],
            "defaultGpuId": None,
        }
        ensure_rife_runtime_mock.return_value = {
            "rifePath": "rife.exe",
            "rifeModelRoot": "models/rife",
        }
        probe_video_mock.return_value = {
            "width": 1280,
            "height": 720,
            "frameRate": 60.0,
            "durationSeconds": 6.0,
            "hasAudio": True,
            "videoCodec": "h264",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            with self.assertRaises(ValueError):
                run_realesrgan_pipeline(
                    source_path="C:/fixtures/input.mp4",
                    model_id="realesrgan-x4plus",
                    output_mode="preserveAspect4k",
                    preset="qualityBalanced",
                    interpolation_mode="interpolateOnly",
                    interpolation_target_fps=30,
                    gpu_id=0,
                    aspect_ratio_preset="16:9",
                    custom_aspect_width=None,
                    custom_aspect_height=None,
                    resolution_basis="exact",
                    target_width=3840,
                    target_height=2160,
                    crop_left=None,
                    crop_top=None,
                    crop_width=None,
                    crop_height=None,
                    progress_path=str(temp_root / "progress.json"),
                    cancel_path=None,
                    preview_mode=False,
                    preview_duration_seconds=None,
                    segment_duration_seconds=10.0,
                    output_path=str(temp_root / "output.mp4"),
                    codec="h264",
                    container="mp4",
                    tile_size=0,
                    fp16=False,
                    torch_compile_enabled=False,
                    crf=18,
                )


if __name__ == "__main__":
    unittest.main()