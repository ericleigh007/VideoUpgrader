import unittest

from upscaler_worker.model_catalog import ensure_runnable_model, ensure_supported_model
from upscaler_worker.models.realesrgan import model_label


class ModelRegistryTests(unittest.TestCase):
    def test_photo_model_label_is_stable(self) -> None:
        self.assertEqual(model_label("realesrgan-x4plus"), "Real-ESRGAN x4 Plus")

    def test_hidden_compatibility_models_remain_supported(self) -> None:
        self.assertEqual(ensure_supported_model("realesrgan-x4plus-anime"), "realesrgan-x4plus-anime")
        self.assertEqual(ensure_supported_model("realesr-animevideov3-x4"), "realesr-animevideov3-x4")

    def test_planned_ranked_models_are_cataloged_but_not_runnable_yet(self) -> None:
        self.assertEqual(ensure_supported_model("hat-realhat-gan-x4"), "hat-realhat-gan-x4")

        with self.assertRaises(ValueError):
            ensure_runnable_model("hat-realhat-gan-x4")

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ensure_supported_model("future-backend-model")


if __name__ == "__main__":
    unittest.main()