from __future__ import annotations

from pathlib import Path

from upscaler_worker.model_catalog import ensure_runnable_model


DDCOLOR_MODEL_IDS = {"ddcolor-modelscope", "ddcolor-paper"}
DEOLDIFY_MODEL_IDS = {"deoldify-stable", "deoldify-video"}
DEEPREMASTER_MODEL_IDS = {"deepremaster"}
COLORMNET_MODEL_IDS = {"colormnet"}


def load_runtime_colorizer(
    model_id: str,
    gpu_id: int | None,
    precision: str | None,
    log: list[str],
    reference_image_paths: list[str] | None = None,
    deepremaster_processing_mode: str = "standard",
):
    ensure_runnable_model(model_id)
    if model_id in DDCOLOR_MODEL_IDS:
        from upscaler_worker.models.ddcolor import load_runtime_colorizer as load_ddcolor_runtime_colorizer

        return load_ddcolor_runtime_colorizer(model_id, gpu_id, precision, log, reference_image_paths=reference_image_paths)
    if model_id in DEOLDIFY_MODEL_IDS:
        from upscaler_worker.models.deoldify import load_runtime_colorizer as load_deoldify_runtime_colorizer

        return load_deoldify_runtime_colorizer(model_id, gpu_id, precision, log, reference_image_paths=reference_image_paths)
    if model_id in DEEPREMASTER_MODEL_IDS:
        from upscaler_worker.models.deepremaster import load_runtime_colorizer as load_deepremaster_runtime_colorizer

        return load_deepremaster_runtime_colorizer(
            model_id,
            gpu_id,
            precision,
            log,
            reference_image_paths=reference_image_paths,
            processing_mode=deepremaster_processing_mode,
        )
    if model_id in COLORMNET_MODEL_IDS:
        from upscaler_worker.models.colormnet import load_runtime_colorizer as load_colormnet_runtime_colorizer

        return load_colormnet_runtime_colorizer(model_id, gpu_id, precision, log, reference_image_paths=reference_image_paths)
    raise NotImplementedError(f"Colorizer '{model_id}' is not implemented")


def colorize_directory(
    *,
    loaded_model,
    input_dir: Path,
    output_dir: Path,
    cancel_path: str | None,
    pause_path: str | None,
    progress_callback=None,
) -> int:
    if getattr(loaded_model, "model_id", None) in DDCOLOR_MODEL_IDS:
        from upscaler_worker.models.ddcolor import colorize_directory as ddcolorize_directory

        return ddcolorize_directory(
            loaded_model=loaded_model,
            input_dir=input_dir,
            output_dir=output_dir,
            cancel_path=cancel_path,
            pause_path=pause_path,
            progress_callback=progress_callback,
        )
    if getattr(loaded_model, "model_id", None) in DEOLDIFY_MODEL_IDS:
        from upscaler_worker.models.deoldify import colorize_directory as deoldify_colorize_directory

        return deoldify_colorize_directory(
            loaded_model=loaded_model,
            input_dir=input_dir,
            output_dir=output_dir,
            cancel_path=cancel_path,
            pause_path=pause_path,
            progress_callback=progress_callback,
        )
    if getattr(loaded_model, "model_id", None) in DEEPREMASTER_MODEL_IDS:
        from upscaler_worker.models.deepremaster import colorize_directory as deepremaster_colorize_directory

        return deepremaster_colorize_directory(
            loaded_model=loaded_model,
            input_dir=input_dir,
            output_dir=output_dir,
            cancel_path=cancel_path,
            pause_path=pause_path,
            progress_callback=progress_callback,
        )
    if getattr(loaded_model, "model_id", None) in COLORMNET_MODEL_IDS:
        from upscaler_worker.models.colormnet import colorize_directory as colormnet_colorize_directory

        return colormnet_colorize_directory(
            loaded_model=loaded_model,
            input_dir=input_dir,
            output_dir=output_dir,
            cancel_path=cancel_path,
            pause_path=pause_path,
            progress_callback=progress_callback,
        )
    raise NotImplementedError(f"Loaded colorizer '{getattr(loaded_model, 'model_id', '<unknown>')}' is not implemented")