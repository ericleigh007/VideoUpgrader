import unittest
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as F
    from upscaler_worker.models.pytorch_sr import _load_array_batch, _maybe_enable_channels_last, _maybe_enable_torch_compile, _run_descriptor, _select_frame_batch_size, resolve_frame_batch_size_override, resolve_precision_mode, resolve_pytorch_runner, resolve_torch_compile_mode
except ModuleNotFoundError:
    torch = None
    F = None
    _load_array_batch = None
    _maybe_enable_channels_last = None
    _maybe_enable_torch_compile = None
    _run_descriptor = None
    _select_frame_batch_size = None
    resolve_frame_batch_size_override = None
    resolve_precision_mode = None
    resolve_pytorch_runner = None
    resolve_torch_compile_mode = None


@unittest.skipUnless(torch is not None, "torch-backed tests require the worker runtime dependencies")
class PytorchSrTests(unittest.TestCase):
    def test_maybe_enable_channels_last_skips_when_disabled(self) -> None:
        class Descriptor:
            def __init__(self) -> None:
                self.model = torch.nn.Identity()

        descriptor = Descriptor()
        self.assertFalse(_maybe_enable_channels_last(descriptor, torch.device("cuda:0"), [], False))

    def test_maybe_enable_torch_compile_skips_when_disabled(self) -> None:
        class Descriptor:
            def __init__(self) -> None:
                self.model = torch.nn.Identity()

        descriptor = Descriptor()
        self.assertFalse(_maybe_enable_torch_compile(descriptor, torch.device("cuda:0"), [], False))

    def test_select_frame_batch_size_is_conservative_on_cpu(self) -> None:
        self.assertEqual(_select_frame_batch_size(torch.device("cpu"), tile_size=128, dtype=torch.float16), 1)

    @patch("upscaler_worker.models.pytorch_sr._query_cuda_memory_bytes", return_value=(24 * 1024**3, 24 * 1024**3))
    def test_select_frame_batch_size_scales_for_cuda_tile_sizes(self, *_mocks) -> None:
        self.assertEqual(_select_frame_batch_size(torch.device("cuda:0"), tile_size=128, dtype=torch.float16), 8)
        self.assertEqual(_select_frame_batch_size(torch.device("cuda:0"), tile_size=256, dtype=torch.float16), 4)
        self.assertEqual(_select_frame_batch_size(torch.device("cuda:0"), tile_size=384, dtype=torch.float32), 1)

    @patch("upscaler_worker.models.pytorch_sr._query_cuda_memory_bytes", return_value=(24 * 1024**3, 24 * 1024**3))
    def test_select_frame_batch_size_honors_vram_safe_preset(self, *_mocks) -> None:
        self.assertEqual(
            _select_frame_batch_size(torch.device("cuda:0"), tile_size=128, dtype=torch.float16, preset="vramSafe"),
            2,
        )

    @patch("upscaler_worker.models.pytorch_sr._query_cuda_memory_bytes", return_value=(24 * 1024**3, 24 * 1024**3))
    def test_select_frame_batch_size_treats_bf16_like_fp16(self, *_mocks) -> None:
        self.assertEqual(_select_frame_batch_size(torch.device("cuda:0"), tile_size=256, dtype=torch.bfloat16), 4)

    def test_resolve_precision_mode_accepts_matching_legacy_flag(self) -> None:
        self.assertEqual(resolve_precision_mode(fp16=False, bf16=True, precision="bf16"), "bf16")

    def test_resolve_precision_mode_rejects_conflicting_legacy_flag(self) -> None:
        with self.assertRaises(ValueError):
            resolve_precision_mode(fp16=True, bf16=False, precision="bf16")

    def test_resolve_torch_compile_mode_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            resolve_torch_compile_mode("turbo")

    def test_resolve_pytorch_runner_rejects_unknown_runner(self) -> None:
        with self.assertRaises(ValueError):
            resolve_pytorch_runner("directml")

    def test_resolve_pytorch_runner_defaults_to_torch(self) -> None:
        self.assertEqual(resolve_pytorch_runner(None), "torch")

    def test_resolve_frame_batch_size_override_accepts_positive_integer(self) -> None:
        self.assertEqual(resolve_frame_batch_size_override("6"), 6)

    def test_resolve_frame_batch_size_override_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            resolve_frame_batch_size_override("0")
        with self.assertRaises(ValueError):
            resolve_frame_batch_size_override("abc")

    @patch("torch.compile")
    def test_maybe_enable_torch_compile_wraps_descriptor_model(self, compile_mock) -> None:
        class Descriptor:
            def __init__(self) -> None:
                self.model = torch.nn.Identity()

        descriptor = Descriptor()
        compiled = torch.nn.Identity()
        compile_mock.return_value = compiled
        log: list[str] = []

        enabled = _maybe_enable_torch_compile(descriptor, torch.device("cuda:0"), log, True, mode="max-autotune", cudagraphs=True)

        self.assertTrue(enabled)
        self.assertIs(descriptor._model, compiled)
        compile_mock.assert_called_once_with(descriptor.model, mode="max-autotune", fullgraph=False)
        self.assertTrue(any("Enabled torch.compile" in entry for entry in log))

    def test_run_descriptor_supports_batched_non_tiled_input(self) -> None:
        image_tensor = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

        def descriptor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor + 1

        output = _run_descriptor(descriptor, image_tensor, tile_size=0, scale=1)
        self.assertEqual(output.shape, image_tensor.shape)
        self.assertTrue(torch.equal(output, image_tensor + 1))

    def test_load_array_batch_can_use_channels_last(self) -> None:
        arrays = [torch.zeros((4, 4, 3), dtype=torch.float32).numpy()]

        tensor = _load_array_batch(arrays, torch.device("cpu"), torch.float32, False, channels_last=True)

        self.assertTrue(tensor.is_contiguous(memory_format=torch.channels_last))

    def test_run_descriptor_supports_batched_tiled_input(self) -> None:
        image_tensor = torch.rand((2, 3, 5, 6), dtype=torch.float32)

        def descriptor(tensor: torch.Tensor) -> torch.Tensor:
            return F.interpolate(tensor, scale_factor=2, mode="nearest")

        output = _run_descriptor(descriptor, image_tensor, tile_size=3, scale=2)
        expected = descriptor(image_tensor).cpu().float()
        self.assertEqual(output.shape, expected.shape)
        self.assertTrue(torch.allclose(output, expected))


if __name__ == "__main__":
    unittest.main()