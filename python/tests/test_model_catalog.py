import unittest

from upscaler_worker.model_catalog import (
    comparison_eligible_models,
    ensure_benchmarkable_model,
    ensure_runnable_model,
    get_backend_definition,
    get_model_definition,
    model_backend_id,
    model_execution_status,
    model_loader,
    model_native_scale,
    model_research_runtime,
    model_runtime_name,
    model_special_handling,
    top_rated_models,
    visible_ui_models,
)


class ModelCatalogTests(unittest.TestCase):
    def test_visible_ui_models_include_runnable_serious_candidates(self) -> None:
        visible_models = visible_ui_models()

        self.assertEqual(
            [model["id"] for model in visible_models],
            [
                "realesrgan-x4plus",
                "realesrnet-x4plus",
                "bsrgan-x4",
                "swinir-realworld-x4",
                "rvrt-x4",
            ],
        )

    def test_top_rated_models_cover_the_selected_six_model_strategy(self) -> None:
        ranked_models = top_rated_models()

        self.assertEqual(
            [model["id"] for model in ranked_models],
            [
                "realesrgan-x4plus",
                "realesrnet-x4plus",
                "bsrgan-x4",
                "hat-realhat-gan-x4",
                "swinir-realworld-x4",
                "rvrt-x4",
            ],
        )

    def test_comparison_eligible_models_only_include_currently_runnable_serious_entries(self) -> None:
        compare_ready = comparison_eligible_models()

        self.assertEqual(
            [model["id"] for model in compare_ready],
            [
                "realesrgan-x4plus",
                "realesrnet-x4plus",
                "bsrgan-x4",
                "swinir-realworld-x4",
            ],
        )

    def test_photo_model_exposes_backend_and_loader_metadata(self) -> None:
        model = get_model_definition("realesrgan-x4plus")
        backend = get_backend_definition(model_backend_id("realesrgan-x4plus"))

        self.assertEqual(model["label"], "Real-ESRGAN x4 Plus")
        self.assertEqual(model_loader("realesrgan-x4plus"), "ncnn-portable-binary")
        self.assertEqual(model_runtime_name("realesrgan-x4plus"), "realesrgan-x4plus")
        self.assertEqual(model_native_scale("realesrgan-x4plus"), 4)
        self.assertEqual(backend["id"], "realesrgan-ncnn")
        self.assertTrue(bool(backend["supportsGpuSelection"]))

    def test_variant_suffix_flag_is_available_for_scale_specific_models(self) -> None:
        handling = model_special_handling("realesr-animevideov3-x4")

        self.assertEqual(handling["cropMath"], "scale-aware")
        self.assertTrue(bool(handling["variantSuffixByScale"]))

    def test_planned_model_is_not_marked_runnable(self) -> None:
        self.assertEqual(model_execution_status("hat-realhat-gan-x4"), "planned")

        with self.assertRaises(ValueError):
            ensure_runnable_model("hat-realhat-gan-x4")

    def test_research_video_model_is_runnable_through_external_runner_contract(self) -> None:
        self.assertEqual(model_execution_status("rvrt-x4"), "runnable")
        self.assertEqual(ensure_runnable_model("rvrt-x4"), "rvrt-x4")
        self.assertEqual(ensure_benchmarkable_model("rvrt-x4"), "rvrt-x4")
        self.assertEqual(model_research_runtime("rvrt-x4")["commandEnvVar"], "UPSCALER_RVRT_COMMAND")

    def test_non_research_planned_model_is_not_benchmarkable(self) -> None:
        with self.assertRaises(ValueError):
            ensure_benchmarkable_model("hat-realhat-gan-x4")


if __name__ == "__main__":
    unittest.main()