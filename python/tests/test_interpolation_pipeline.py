import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from upscaler_worker.interpolation import resolve_segment_output_frame_count
from upscaler_worker.pipeline import (
    PipelineProgressState,
    PipelineSegment,
    _encode_segment_video,
    _find_regular_tile_seams,
    _plan_interpolation_segments,
    _resolve_realesrgan_ncnn_scale,
    _should_publish_stage_progress,
    _validate_segment_visual_integrity,
    run_realesrgan_pipeline,
)


class InterpolationPipelineTests(unittest.TestCase):
    def test_regular_tile_seam_detector_flags_block_corruption(self) -> None:
        frame = np.full((180, 320), 96, dtype=np.float32)
        frame[:60, :160] = 40
        frame[:60, 160:] = 180
        frame[60:120, :160] = 150
        frame[60:120, 160:] = 70
        frame[120:, :160] = 210
        frame[120:, 160:] = 30

        seams = _find_regular_tile_seams(frame)

        self.assertTrue(any(seam.startswith("row") for seam in seams))
        self.assertTrue(any(seam.startswith("col") for seam in seams))

    def test_regular_tile_seam_detector_ignores_smooth_gradient(self) -> None:
        horizontal = np.linspace(0, 255, 320, dtype=np.float32)
        frame = np.tile(horizontal, (180, 1))

        self.assertEqual(_find_regular_tile_seams(frame), [])

    def test_segment_visual_validation_rejects_decode_failure(self) -> None:
        log: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            segment_file = Path(temp_dir) / "segment_0000.mkv"
            segment_file.write_bytes(b"not a video")
            with patch("upscaler_worker.pipeline.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
                self.assertFalse(_validate_segment_visual_integrity("ffmpeg", segment_file, log))

        self.assertIn("could not decode validation frame", "\n".join(log))

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress")
    def test_segment_encode_promotes_part_file_after_success(self, ffmpeg_progress_mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "segment_0000.mkv"
            part_file = output_file.with_name("segment_0000.part.mkv")

            def write_part_file(*_args, **_kwargs):
                part_file.write_bytes(b"encoded")
                return 24, None

            ffmpeg_progress_mock.side_effect = write_part_file

            with patch("upscaler_worker.pipeline._validate_segment_visual_integrity", return_value=True):
                encoded_frames = _encode_segment_video(
                    ffmpeg="ffmpeg",
                    upscaled_dir=Path(temp_dir),
                    output_file=output_file,
                    fps="24.000",
                    codec="h264",
                    crf=18,
                    video_encoder_config=SimpleNamespace(encoder="libx264", quality_args=()),
                    filter_chain=None,
                    model_name="realesrgan-x4plus",
                    output_mode="preserveAspect4k",
                    aspect_ratio_preset="16:9",
                    resolution_basis="exact",
                    resolved_width=3840,
                    resolved_height=2160,
                    crop_left=None,
                    crop_top=None,
                    crop_width=None,
                    crop_height=None,
                    container="mkv",
                    log=[],
                    progress_path=None,
                    cancel_path=None,
                    pause_path=None,
                    total_frames=24,
                    extracted_frames=24,
                    colorized_frames=0,
                    upscaled_frames=24,
                    interpolated_frames=0,
                    encoded_frames_before_segment=0,
                    telemetry_state=None,
                )

            self.assertEqual(encoded_frames, 24)
            self.assertTrue(output_file.exists())
            self.assertFalse(part_file.exists())
            self.assertEqual(output_file.read_bytes(), b"encoded")
            self.assertEqual(ffmpeg_progress_mock.call_args.args[0][-1], str(part_file))

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress")
    @patch("upscaler_worker.pipeline._validate_segment_visual_integrity")
    def test_segment_encode_skips_visual_validation_when_disabled(
        self,
        validate_mock,
        ffmpeg_progress_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "segment_0000.mkv"
            part_file = output_file.with_name("segment_0000.part.mkv")

            def write_part_file(*_args, **_kwargs):
                part_file.write_bytes(b"encoded")
                return 24, None

            ffmpeg_progress_mock.side_effect = write_part_file
            validate_mock.return_value = False

            encoded_frames = _encode_segment_video(
                ffmpeg="ffmpeg",
                upscaled_dir=Path(temp_dir),
                output_file=output_file,
                fps="24.000",
                codec="h264",
                crf=18,
                video_encoder_config=SimpleNamespace(encoder="libx264", quality_args=()),
                filter_chain=None,
                model_name="realesrgan-x4plus",
                output_mode="preserveAspect4k",
                aspect_ratio_preset="16:9",
                resolution_basis="exact",
                resolved_width=3840,
                resolved_height=2160,
                crop_left=None,
                crop_top=None,
                crop_width=None,
                crop_height=None,
                container="mkv",
                log=[],
                progress_path=None,
                cancel_path=None,
                pause_path=None,
                total_frames=24,
                extracted_frames=24,
                colorized_frames=0,
                upscaled_frames=24,
                interpolated_frames=0,
                encoded_frames_before_segment=0,
                telemetry_state=None,
                validate_visual_integrity=False,
            )

            self.assertEqual(encoded_frames, 24)
            self.assertTrue(output_file.exists())
            validate_mock.assert_not_called()

    def test_extracting_progress_is_suppressed_after_downstream_stage_starts(self) -> None:
        progress_state = PipelineProgressState(extracted_frames=2396, upscaled_frames=128)

        self.assertFalse(_should_publish_stage_progress("extracting", progress_state))
        self.assertTrue(_should_publish_stage_progress("upscaling", progress_state))
        self.assertTrue(_should_publish_stage_progress("encoding", progress_state))

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

    def test_resolve_realesrgan_ncnn_scale_matches_preserve_aspect_target(self) -> None:
        self.assertEqual(
            _resolve_realesrgan_ncnn_scale(
                output_mode="preserveAspect4k",
                source_width=966,
                source_height=720,
                target_width=3840,
                target_height=2160,
            ),
            3,
        )
        self.assertEqual(
            _resolve_realesrgan_ncnn_scale(
                output_mode="cropTo4k",
                source_width=966,
                source_height=720,
                target_width=3840,
                target_height=2160,
            ),
            4,
        )

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", side_effect=[(12, 12), (12, 12)])
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_backend_id", return_value="realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    @patch("upscaler_worker.pipeline._resolve_video_encoder_config")
    @patch("upscaler_worker.pipeline._extract_segment_frames", return_value=12)
    @patch("upscaler_worker.pipeline._upscale_ncnn_segment", return_value=12)
    @patch("upscaler_worker.pipeline._encode_segment_video", return_value=12)
    def test_pipeline_uses_explicit_job_id_for_workdir(
        self,
        _encode_mock,
        _upscale_mock,
        _extract_mock,
        resolve_video_encoder_config_mock,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        ensure_runtime_assets_mock,
        probe_video_mock,
        _ffmpeg_progress_mock,
    ) -> None:
        ensure_runtime_assets_mock.return_value = {
            "ffmpegPath": "ffmpeg",
            "realesrganPath": "realesrgan.exe",
            "modelDir": "models/realesrgan",
            "availableGpus": [],
            "defaultGpuId": None,
        }
        resolve_video_encoder_config_mock.return_value = type(
            "EncoderConfig",
            (),
            {"encoder": "libx264", "quality_args": (), "label": "software-cpu", "hardware_accelerated": False},
        )()
        probe_video_mock.return_value = {
            "width": 320,
            "height": 180,
            "frameRate": 24.0,
            "durationSeconds": 0.5,
            "hasAudio": False,
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
                interpolation_mode="off",
                interpolation_target_fps=None,
                gpu_id=0,
                aspect_ratio_preset="16:9",
                custom_aspect_width=None,
                custom_aspect_height=None,
                resolution_basis="exact",
                target_width=640,
                target_height=360,
                crop_left=None,
                crop_top=None,
                crop_width=None,
                crop_height=None,
                job_id="canonical-job-id",
                progress_path=str(progress_path),
                cancel_path=None,
                preview_mode=True,
                preview_duration_seconds=0.5,
                segment_duration_seconds=None,
                output_path=str(output_path),
                codec="h264",
                container="mp4",
                tile_size=0,
                fp16=False,
                torch_compile_enabled=False,
                crf=18,
            )

            self.assertTrue(str(result["workDir"]).endswith("job_canonical-job-id"))
            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["jobId"], "canonical-job-id")
            self.assertEqual(progress_payload["sourcePath"], "C:/fixtures/input.mp4")
            self.assertEqual(progress_payload["outputPath"], str(output_path))
            self.assertTrue(str(progress_payload["scratchPath"]).endswith("job_canonical-job-id"))

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", return_value=(300, 300))
    @patch("upscaler_worker.pipeline._concat_segment_videos", return_value=300)
    @patch("upscaler_worker.pipeline._encode_segment_video", return_value=60)
    @patch("upscaler_worker.pipeline._upscale_ncnn_segment", return_value=60)
    @patch("upscaler_worker.pipeline._extract_segment_frames", return_value=60)
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_backend_id", return_value="realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    @patch("upscaler_worker.pipeline._resolve_video_encoder_config")
    def test_segmented_pipeline_resume_skips_completed_segment_checkpoint(
        self,
        resolve_video_encoder_config_mock,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        ensure_runtime_assets_mock,
        probe_video_mock,
        extract_mock,
        upscale_mock,
        encode_mock,
        concat_mock,
        _ffmpeg_progress_mock,
    ) -> None:
        ensure_runtime_assets_mock.return_value = {
            "ffmpegPath": "ffmpeg",
            "realesrganPath": "realesrgan.exe",
            "modelDir": "models/realesrgan",
            "availableGpus": [],
            "defaultGpuId": None,
        }
        resolve_video_encoder_config_mock.return_value = type(
            "EncoderConfig",
            (),
            {"encoder": "libx264", "quality_args": (), "label": "software-cpu", "hardware_accelerated": False},
        )()
        probe_video_mock.return_value = {
            "width": 320,
            "height": 180,
            "frameRate": 24.0,
            "durationSeconds": 12.5,
            "hasAudio": False,
            "videoCodec": "h264",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            checkpoint_dir = temp_root / "artifacts" / "jobs" / "job_resume-job" / "enc"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_segment = checkpoint_dir / "segment_0000.mkv"
            checkpoint_segment.write_bytes(b"completed segment")
            output_path = temp_root / "output.mp4"
            progress_path = temp_root / "progress.json"

            with patch("upscaler_worker.pipeline.repo_root", return_value=temp_root), patch(
                "upscaler_worker.pipeline._validate_segment_visual_integrity", return_value=True
            ):
                result = run_realesrgan_pipeline(
                    source_path="C:/fixtures/input.mp4",
                    model_id="realesrgan-x4plus",
                    output_mode="preserveAspect4k",
                    preset="qualityBalanced",
                    interpolation_mode="off",
                    interpolation_target_fps=None,
                    gpu_id=0,
                    aspect_ratio_preset="16:9",
                    custom_aspect_width=None,
                    custom_aspect_height=None,
                    resolution_basis="exact",
                    target_width=640,
                    target_height=360,
                    crop_left=None,
                    crop_top=None,
                    crop_width=None,
                    crop_height=None,
                    job_id="resume-job",
                    resume_from_job_id="resume-job",
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

            self.assertEqual(result["segmentCount"], 2)
            self.assertIn("Resumed from checkpoint: encoded segment 1/2", "\n".join(result["log"]))
            self.assertEqual(extract_mock.call_count, 1)
            self.assertEqual(upscale_mock.call_count, 1)
            self.assertEqual(encode_mock.call_count, 1)
            concat_mock.assert_called_once()
            self.assertEqual(concat_mock.call_args.kwargs["segment_files"][0], checkpoint_segment)

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
            for call in upscale_mock.call_args_list:
                self.assertEqual(call.kwargs["upscale_scale"], 3)
            self.assertEqual(rife_mock.call_count, 2)
            self.assertEqual(encode_mock.call_count, 2)
            concat_mock.assert_called_once()
            self.assertEqual(ffmpeg_progress_mock.call_count, 1)

            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["phase"], "completed")
            self.assertEqual(progress_payload["totalFrames"], 750)
            self.assertEqual(progress_payload["interpolatedFrames"], 750)
            self.assertEqual(progress_payload["remuxedFrames"], 750)

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", return_value=(750, 750))
    @patch("upscaler_worker.pipeline._concat_segment_videos", return_value=750)
    @patch("upscaler_worker.pipeline._encode_segment_video", return_value=150)
    @patch("upscaler_worker.pipeline._run_rife_segment", return_value=153)
    @patch("upscaler_worker.pipeline._upscale_ncnn_segment", return_value=61)
    @patch("upscaler_worker.pipeline._extract_segment_frames", return_value=61)
    @patch("upscaler_worker.pipeline.ensure_rife_runtime")
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_backend_id", return_value="realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    def test_after_upscale_interpolation_resume_skips_completed_segment_checkpoint(
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
        _ffmpeg_progress_mock,
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
            "hasAudio": False,
            "videoCodec": "h264",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            checkpoint_dir = temp_root / "artifacts" / "jobs" / "job_resume-job" / "enc"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_segment = checkpoint_dir / "segment_0000.mkv"
            checkpoint_segment.write_bytes(b"completed segment")
            output_path = temp_root / "output.mp4"
            progress_path = temp_root / "progress.json"

            with patch("upscaler_worker.pipeline.repo_root", return_value=temp_root), patch(
                "upscaler_worker.pipeline._validate_segment_visual_integrity", return_value=True
            ):
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
                    job_id="resume-job",
                    resume_from_job_id="resume-job",
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
            self.assertIn("Resumed from checkpoint: encoded segment 1/2", "\n".join(result["log"]))
            self.assertEqual(extract_mock.call_count, 1)
            self.assertEqual(upscale_mock.call_count, 1)
            self.assertEqual(rife_mock.call_count, 1)
            self.assertEqual(encode_mock.call_count, 1)
            concat_mock.assert_called_once()
            self.assertEqual(concat_mock.call_args.kwargs["segment_files"][0], checkpoint_segment)

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", side_effect=[(750, 750), (750, 750)])
    @patch("upscaler_worker.pipeline._concat_segment_videos", return_value=750)
    @patch("upscaler_worker.pipeline._encode_segment_video", side_effect=[600, 150])
    @patch("upscaler_worker.pipeline._run_rife_segment", side_effect=[603, 153])
    @patch("upscaler_worker.pipeline._upscale_ncnn_segment", side_effect=[241, 61])
    @patch("upscaler_worker.pipeline._denoise_segment", side_effect=[241, 61])
    @patch("upscaler_worker.pipeline._extract_segment_frames", side_effect=[241, 61])
    @patch("upscaler_worker.pipeline.ensure_rife_runtime")
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_task", return_value="denoise")
    @patch("upscaler_worker.pipeline.model_backend_id", side_effect=lambda model_id: "pytorch-image-denoise" if model_id == "drunet-gray-color-denoise" else "realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    def test_after_upscale_interpolation_with_denoise_uses_selected_precision(
        self,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        _model_task_mock,
        ensure_runtime_assets_mock,
        probe_video_mock,
        ensure_rife_runtime_mock,
        _extract_mock,
        denoise_mock,
        _upscale_mock,
        _rife_mock,
        _encode_mock,
        _concat_mock,
        _ffmpeg_progress_mock,
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
                denoise_mode="beforeUpscale",
                denoiser_model_id="drunet-gray-color-denoise",
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
                codec="h265",
                container="mp4",
                tile_size=0,
                fp16=False,
                torch_compile_enabled=False,
                crf=18,
            )

            self.assertEqual(result["frameCount"], 750)
            self.assertEqual(denoise_mock.call_count, 2)
            for call in denoise_mock.call_args_list:
                self.assertEqual(call.kwargs["precision"], "bf16")

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
            overlap_verified["value"] = first_encode_started.wait(timeout=2.0)
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

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", side_effect=[(30, 30), (30, 30)])
    @patch("upscaler_worker.pipeline._concat_segment_videos", return_value=30)
    @patch("upscaler_worker.pipeline._encode_segment_video", return_value=30)
    @patch("upscaler_worker.pipeline._run_rife_segment", return_value=30)
    @patch("upscaler_worker.pipeline._extract_segment_frames", return_value=12)
    @patch("upscaler_worker.pipeline.ensure_rife_runtime")
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_task", side_effect=lambda model_id: "colorize" if model_id == "ddcolor-modelscope" else "upscale")
    @patch("upscaler_worker.pipeline.model_backend_id", side_effect=lambda model_id: "pytorch-image-colorization" if model_id == "ddcolor-modelscope" else "realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    @patch("upscaler_worker.models.colorizers.colorize_directory", return_value=12)
    @patch("upscaler_worker.models.colorizers.load_runtime_colorizer")
    def test_colorize_only_interpolation_pipeline_runs_colorizer_before_rife(
        self,
        load_runtime_colorizer_mock,
        colorize_directory_mock,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        _model_task_mock,
        ensure_runtime_assets_mock,
        probe_video_mock,
        ensure_rife_runtime_mock,
        extract_mock,
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
            "width": 320,
            "height": 180,
            "frameRate": 24.0,
            "durationSeconds": 0.5,
            "hasAudio": True,
            "videoCodec": "h264",
        }
        load_runtime_colorizer_mock.return_value = SimpleNamespace(
            precision_mode="fp32",
            repo_id="ddcolor/modelscope",
            checkpoint_path=None,
            input_size=512,
            device=SimpleNamespace(type="cpu"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "output.mp4"
            progress_path = temp_root / "progress.json"

            result = run_realesrgan_pipeline(
                source_path="C:/fixtures/input.mp4",
                model_id="ddcolor-modelscope",
                colorization_mode="colorizeOnly",
                colorizer_model_id="ddcolor-modelscope",
                output_mode="preserveAspect4k",
                preset="qualityBalanced",
                interpolation_mode="interpolateOnly",
                interpolation_target_fps=60,
                gpu_id=0,
                aspect_ratio_preset="16:9",
                custom_aspect_width=None,
                custom_aspect_height=None,
                resolution_basis="exact",
                target_width=320,
                target_height=180,
                crop_left=None,
                crop_top=None,
                crop_width=None,
                crop_height=None,
                progress_path=str(progress_path),
                cancel_path=None,
                preview_mode=True,
                preview_duration_seconds=0.5,
                segment_duration_seconds=None,
                output_path=str(output_path),
                codec="h264",
                container="mp4",
                tile_size=0,
                fp16=False,
                torch_compile_enabled=False,
                crf=18,
            )

            self.assertEqual(result["frameCount"], 30)
            self.assertEqual(result["executionPath"], "rife-ncnn-vulkan")
            self.assertEqual(result["runner"], "torch")
            self.assertIn("colorizeSeconds", result["stageTimings"])
            self.assertIn("Colorizer: DDColor ModelScope (ddcolor-modelscope)", "\n".join(result["log"]))
            self.assertEqual(colorize_directory_mock.call_count, 1)
            self.assertEqual(rife_mock.call_count, 1)
            encode_mock.assert_called_once()
            concat_mock.assert_called_once()
            self.assertEqual(ffmpeg_progress_mock.call_count, 1)
            self.assertEqual(encode_mock.call_args.kwargs["colorized_frames"], 12)

            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["phase"], "completed")
            self.assertEqual(progress_payload["colorizedFrames"], 12)
            self.assertEqual(progress_payload["interpolatedFrames"], 30)

    @patch("upscaler_worker.pipeline._run_ffmpeg_with_frame_progress", side_effect=[(30, 30), (30, 30)])
    @patch("upscaler_worker.pipeline._concat_segment_videos", return_value=30)
    @patch("upscaler_worker.pipeline._encode_segment_video", return_value=30)
    @patch("upscaler_worker.pipeline._run_rife_segment", return_value=30)
    @patch("upscaler_worker.pipeline._upscale_ncnn_segment", return_value=12)
    @patch("upscaler_worker.pipeline._extract_segment_frames", return_value=12)
    @patch("upscaler_worker.pipeline.ensure_rife_runtime")
    @patch("upscaler_worker.pipeline.probe_video")
    @patch("upscaler_worker.pipeline.ensure_runtime_assets")
    @patch("upscaler_worker.pipeline.model_task", side_effect=lambda model_id: "colorize" if model_id == "ddcolor-modelscope" else "upscale")
    @patch("upscaler_worker.pipeline.model_backend_id", side_effect=lambda model_id: "pytorch-image-colorization" if model_id == "ddcolor-modelscope" else "realesrgan-ncnn")
    @patch("upscaler_worker.pipeline.ensure_runnable_model")
    @patch("upscaler_worker.models.colorizers.colorize_directory", return_value=12)
    @patch("upscaler_worker.models.colorizers.load_runtime_colorizer")
    def test_before_upscale_interpolation_pipeline_colorizes_before_upscaling(
        self,
        load_runtime_colorizer_mock,
        colorize_directory_mock,
        _ensure_runnable_model_mock,
        _model_backend_id_mock,
        _model_task_mock,
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
            "width": 320,
            "height": 180,
            "frameRate": 24.0,
            "durationSeconds": 0.5,
            "hasAudio": True,
            "videoCodec": "h264",
        }
        load_runtime_colorizer_mock.return_value = SimpleNamespace(
            precision_mode="fp32",
            repo_id="ddcolor/modelscope",
            checkpoint_path=None,
            input_size=512,
            device=SimpleNamespace(type="cpu"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "output.mp4"
            progress_path = temp_root / "progress.json"

            result = run_realesrgan_pipeline(
                source_path="C:/fixtures/input.mp4",
                model_id="realesrgan-x4plus",
                colorization_mode="beforeUpscale",
                colorizer_model_id="ddcolor-modelscope",
                output_mode="preserveAspect4k",
                preset="qualityBalanced",
                interpolation_mode="afterUpscale",
                interpolation_target_fps=60,
                gpu_id=0,
                aspect_ratio_preset="16:9",
                custom_aspect_width=None,
                custom_aspect_height=None,
                resolution_basis="exact",
                target_width=1280,
                target_height=720,
                crop_left=None,
                crop_top=None,
                crop_width=None,
                crop_height=None,
                progress_path=str(progress_path),
                cancel_path=None,
                preview_mode=True,
                preview_duration_seconds=0.5,
                segment_duration_seconds=None,
                output_path=str(output_path),
                codec="h264",
                container="mp4",
                tile_size=0,
                fp16=False,
                torch_compile_enabled=False,
                crf=18,
            )

            self.assertEqual(result["frameCount"], 30)
            self.assertIn("colorizeSeconds", result["stageTimings"])
            self.assertEqual(colorize_directory_mock.call_count, 1)
            upscale_mock.assert_called_once()
            self.assertEqual(upscale_mock.call_args.kwargs["input_dir"].name, "colorized")
            self.assertEqual(encode_mock.call_args.kwargs["colorized_frames"], 12)
            self.assertIn("Colorizer: DDColor ModelScope (ddcolor-modelscope)", "\n".join(result["log"]))

            progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["phase"], "completed")
            self.assertEqual(progress_payload["colorizedFrames"], 12)
            self.assertEqual(progress_payload["upscaledFrames"], 12)
            self.assertEqual(progress_payload["interpolatedFrames"], 30)


if __name__ == "__main__":
    unittest.main()
