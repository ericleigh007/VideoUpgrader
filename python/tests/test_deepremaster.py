import unittest

from upscaler_worker.models.deepremaster import _resolve_processing_mode, _resolve_processing_size


class DeepRemasterTests(unittest.TestCase):
    def test_processing_mode_resolves_standard_and_high(self) -> None:
        self.assertEqual(_resolve_processing_mode("standard"), ("standard", 320))
        self.assertEqual(_resolve_processing_mode("high"), ("high", 512))

    def test_processing_size_uses_requested_min_edge(self) -> None:
        standard_size = _resolve_processing_size(320, 180, 320)
        high_size = _resolve_processing_size(320, 180, 512)

        self.assertEqual(standard_size, (576, 320))
        self.assertEqual(high_size, (912, 512))


if __name__ == "__main__":
    unittest.main()