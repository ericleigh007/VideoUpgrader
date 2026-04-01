import unittest

try:
    import torch
    import torch.nn.functional as F
    from upscaler_worker.models.pytorch_sr import _run_descriptor, _select_frame_batch_size
except ModuleNotFoundError:
    torch = None
    F = None
    _run_descriptor = None
    _select_frame_batch_size = None


@unittest.skipUnless(torch is not None, "torch-backed tests require the worker runtime dependencies")
class PytorchSrTests(unittest.TestCase):
    def test_select_frame_batch_size_is_conservative_on_cpu(self) -> None:
        self.assertEqual(_select_frame_batch_size(torch.device("cpu"), tile_size=128, dtype=torch.float16), 1)

    def test_select_frame_batch_size_scales_for_cuda_tile_sizes(self) -> None:
        self.assertEqual(_select_frame_batch_size(torch.device("cuda:0"), tile_size=128, dtype=torch.float16), 4)
        self.assertEqual(_select_frame_batch_size(torch.device("cuda:0"), tile_size=256, dtype=torch.float16), 2)
        self.assertEqual(_select_frame_batch_size(torch.device("cuda:0"), tile_size=384, dtype=torch.float32), 1)

    def test_run_descriptor_supports_batched_non_tiled_input(self) -> None:
        image_tensor = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

        def descriptor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor + 1

        output = _run_descriptor(descriptor, image_tensor, tile_size=0, scale=1)
        self.assertEqual(output.shape, image_tensor.shape)
        self.assertTrue(torch.equal(output, image_tensor + 1))

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