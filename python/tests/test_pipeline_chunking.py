import unittest
from pathlib import Path
from unittest.mock import patch

import upscaler_worker.pipeline as pipeline_module

from upscaler_worker.pipeline import (
    PIPELINE_SEGMENT_FRAME_LIMIT,
    PIPELINE_SEGMENT_TARGET_SECONDS,
    PipelineProgressState,
    _effective_tile_size,
    _pipeline_percent,
    _plan_pipeline_segments,
)


class PipelineChunkingTests(unittest.TestCase):
    def test_plan_pipeline_segments_uses_time_based_chunk_size(self) -> None:
        segments = _plan_pipeline_segments(total_frames=720, fps=24.0)
        expected_frame_limit = int(round(24.0 * PIPELINE_SEGMENT_TARGET_SECONDS))

        self.assertEqual(len(segments), 3)
        self.assertGreaterEqual(expected_frame_limit, PIPELINE_SEGMENT_FRAME_LIMIT)
        self.assertTrue(all(segment.frame_count <= expected_frame_limit for segment in segments))
        self.assertEqual(segments[0].frame_count, expected_frame_limit)
        self.assertAlmostEqual(segments[0].duration_seconds, PIPELINE_SEGMENT_TARGET_SECONDS)
        self.assertEqual(segments[-1].start_frame, expected_frame_limit * 2)

    def test_plan_pipeline_segments_can_force_single_segment(self) -> None:
        segments = _plan_pipeline_segments(total_frames=180, fps=24.0, force_single_segment=True)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].frame_count, 180)
        self.assertEqual(segments[0].start_frame, 0)
        self.assertAlmostEqual(segments[0].duration_seconds, 180 / 24.0)

    def test_plan_pipeline_segments_respects_requested_duration(self) -> None:
        segments = _plan_pipeline_segments(total_frames=720, fps=24.0, segment_duration_seconds=20.0)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].frame_count, 480)
        self.assertAlmostEqual(segments[0].duration_seconds, 20.0)

    def test_pipeline_percent_reflects_overlapped_stage_progress(self) -> None:
        progress_state = PipelineProgressState(
            extracted_frames=48,
            upscaled_frames=24,
            encoded_frames=12,
            remuxed_frames=0,
        )

        self.assertEqual(_pipeline_percent(48, progress_state), 49)

    def test_effective_tile_size_is_backend_aware(self) -> None:
        self.assertEqual(_effective_tile_size("realesrgan-x4plus", "qualityBalanced", 0), 256)
        self.assertEqual(_effective_tile_size("realesrnet-x4plus", "qualityBalanced", 0), 384)
        self.assertEqual(_effective_tile_size("realesrnet-x4plus", "vramSafe", 0), 256)
        self.assertEqual(_effective_tile_size("realesrnet-x4plus", "qualityMax", 0), 512)

    @patch("upscaler_worker.pipeline._path_size_bytes", side_effect=[4096, 2048, 4096, 2048])
    @patch("upscaler_worker.pipeline._sample_gpu_memory_bytes", return_value=(512 * 1024 * 1024, 1024 * 1024 * 1024))
    @patch("upscaler_worker.pipeline._current_process_rss_bytes", return_value=256 * 1024 * 1024)
    @patch("upscaler_worker.pipeline.time.time", side_effect=[110.0, 112.0])
    def test_progress_telemetry_reports_fps_eta_and_resources(self, *_mocks) -> None:
        telemetry_state = pipeline_module.PipelineTelemetryState(
            started_at=100.0,
            scratch_path=Path("scratch"),
            output_path=Path("output.mkv"),
        )

        pipeline_module._sample_progress_telemetry(
            telemetry_state,
            processed_frames=50,
            total_frames=100,
        )

        telemetry = pipeline_module._sample_progress_telemetry(
            telemetry_state,
            processed_frames=70,
            total_frames=100,
        )

        self.assertEqual(telemetry["processRssBytes"], 256 * 1024 * 1024)
        self.assertEqual(telemetry["gpuMemoryUsedBytes"], 512 * 1024 * 1024)
        self.assertEqual(telemetry["gpuMemoryTotalBytes"], 1024 * 1024 * 1024)
        self.assertEqual(telemetry["scratchSizeBytes"], 4096)
        self.assertEqual(telemetry["outputSizeBytes"], 2048)
        self.assertAlmostEqual(float(telemetry["averageFramesPerSecond"]), 70 / 12)
        self.assertAlmostEqual(float(telemetry["rollingFramesPerSecond"]), 10.0)
        self.assertAlmostEqual(float(telemetry["estimatedRemainingSeconds"]), 30 / (70 / 12))


if __name__ == "__main__":
    unittest.main()