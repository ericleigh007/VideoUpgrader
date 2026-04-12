from __future__ import annotations

import argparse
import importlib
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch
import torch.nn.functional as F


RVRT_TASK_BY_MODEL = {
    "rvrt-x4": "002_RVRT_videosr_bi_Vimeo_14frames",
}

SUPPORTED_PRECISIONS = {"fp32", "fp16", "bf16"}
RVRT_TEMPORAL_TILE_BY_MODEL = {
    "rvrt-x4": 14,
}
RVRT_TEMPORAL_TILE_OVERLAP = 2


def _sequence_frames(source_dir: Path) -> list[Path]:
    frames = sorted(source_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No frame_*.png inputs found in '{source_dir}'")
    return frames


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_temporal_tile_size(model_id: str, frame_count: int) -> int:
    configured = RVRT_TEMPORAL_TILE_BY_MODEL.get(model_id, 0)
    if configured <= 0:
        return 0
    if frame_count <= 0:
        return configured
    return min(configured, frame_count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rvrt_external_runner")
    parser.add_argument("--input", required=True, dest="input_dir")
    parser.add_argument("--output", required=True, dest="output_dir")
    parser.add_argument("--model", required=True, dest="model_id")
    parser.add_argument("--tile", type=int, default=128, dest="tile_size")
    parser.add_argument("--gpu-id", type=int, default=None, dest="gpu_id")
    parser.add_argument("--precision", choices=sorted(SUPPORTED_PRECISIONS), default=None)
    return parser


def _resolve_precision(cli_precision: str | None) -> str:
    precision = (cli_precision or os.environ.get("UPSCALER_VIDEO_SR_PRECISION") or "fp32").strip().lower()
    if precision not in SUPPORTED_PRECISIONS:
        raise RuntimeError(f"Unsupported RVRT precision '{precision}'")
    return precision


def _resolve_gpu_id(cli_gpu_id: int | None) -> int | None:
    if cli_gpu_id is not None:
        return cli_gpu_id

    raw_gpu_id = os.environ.get("UPSCALER_VIDEO_SR_GPU_ID", "").strip()
    if not raw_gpu_id:
        return None

    try:
        return int(raw_gpu_id)
    except ValueError as error:
        raise RuntimeError(f"Unsupported RVRT GPU id '{raw_gpu_id}'") from error


def _resolve_device(gpu_id: int | None) -> tuple[torch.device, list[str]]:
    notes: list[str] = []
    if not torch.cuda.is_available():
        return torch.device("cpu"), notes

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return torch.device("cpu"), notes

    if gpu_id is None:
        return torch.device("cuda:0"), notes

    if 0 <= gpu_id < device_count:
        return torch.device(f"cuda:{gpu_id}"), notes

    if device_count == 1 and gpu_id >= 0:
        notes.append(f"Mapped app GPU id {gpu_id} to PyTorch cuda:0.")
        return torch.device("cuda:0"), notes

    notes.append(f"Requested GPU {gpu_id} is not available to PyTorch. Using cuda:0 instead.")
    return torch.device("cuda:0"), notes


def _load_rvrt_modules(rvrt_root: Path):
    rvrt_root_str = str(rvrt_root)
    runner_dir = str(Path(__file__).resolve().parent)
    sys.path = [entry for entry in sys.path if entry and Path(entry).resolve() != Path(runner_dir).resolve()]
    if rvrt_root_str not in sys.path:
        sys.path.insert(0, rvrt_root_str)
    importlib.invalidate_caches()

    for module_name in ["models", "main_test_rvrt", "utils", "utils.utils_video"]:
        sys.modules.pop(module_name, None)

    network_rvrt = importlib.import_module("models.network_rvrt")
    _patch_rvrt_fp16_compatibility(network_rvrt)
    main_test_rvrt = importlib.import_module("main_test_rvrt")
    utils_video = importlib.import_module("utils.utils_video")

    return main_test_rvrt.prepare_model_dataset, main_test_rvrt.test_video, utils_video


def _patch_rvrt_fp16_compatibility(network_rvrt_module) -> None:
    def flow_warp_dtype_safe(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
        n, _, h, w = x.size()
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, dtype=x.dtype, device=x.device),
            torch.arange(0, w, dtype=x.dtype, device=x.device),
        )
        grid = torch.stack((grid_x, grid_y), 2).to(dtype=x.dtype)
        grid.requires_grad = False

        vgrid = grid + flow.to(dtype=x.dtype)
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        return F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    network_rvrt_module.flow_warp = flow_warp_dtype_safe


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision in {"fp32", "fp16"}:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _apply_precision(model: torch.nn.Module, lq: torch.Tensor, precision: str, device: torch.device) -> tuple[torch.nn.Module, torch.Tensor]:
    if precision == "fp32":
        return model.float(), lq.float()
    if device.type != "cuda":
        raise RuntimeError(f"RVRT precision '{precision}' requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return model.to(dtype=dtype), lq.to(dtype=dtype)


def _save_output_frames(*, output: torch.Tensor, output_dir: Path, frame_names: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output.shape[1] != len(frame_names):
        raise RuntimeError(f"Expected {len(frame_names)} output frames, received {output.shape[1]}")

    for frame_index, frame_name in enumerate(frame_names):
        image = output[:, frame_index, ...].detach().float().cpu().clamp_(0, 1).squeeze(0).numpy()
        image = image[[2, 1, 0], :, :].transpose(1, 2, 0)
        image = (image * 255.0).round().astype("uint8")
        cv2.imwrite(str(output_dir / frame_name), image)


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _resolve_repo_root()
    rvrt_root = repo_root / "tmp" / "RVRT"
    if not rvrt_root.exists():
        raise RuntimeError(f"Expected official RVRT repo at '{rvrt_root}'")

    task_name = RVRT_TASK_BY_MODEL.get(args.model_id)
    if task_name is None:
        raise RuntimeError(f"Unsupported RVRT benchmark model '{args.model_id}'")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    precision = _resolve_precision(args.precision)
    gpu_id = _resolve_gpu_id(args.gpu_id)
    frame_paths = _sequence_frames(input_dir)
    prepare_model_dataset, test_video, utils_video = _load_rvrt_modules(rvrt_root)
    temporal_tile_size = _resolve_temporal_tile_size(args.model_id, len(frame_paths))

    rvrt_args = SimpleNamespace(
        task=task_name,
        sigma=0,
        folder_lq=str(input_dir),
        folder_gt=None,
        tile=[temporal_tile_size, max(0, args.tile_size), max(0, args.tile_size)],
        tile_overlap=[RVRT_TEMPORAL_TILE_OVERLAP, 20, 20],
        num_workers=0,
        save_result=False,
    )
    device, device_notes = _resolve_device(gpu_id)
    if device.type != "cuda":
        raise RuntimeError("RVRT external runner requires CUDA on this workstation")

    torch.cuda.set_device(device)
    print(f"RVRT using {device} ({torch.cuda.get_device_name(device)})")
    print(
        f"RVRT tiling temporal={temporal_tile_size or 'full-clip'} "
        f"spatial={max(0, args.tile_size)} precision={precision}"
    )
    for note in device_notes:
        print(note)

    torch.backends.cudnn.benchmark = True
    model = prepare_model_dataset(rvrt_args)
    model.eval()
    model = model.to(device)

    lq = utils_video.read_img_seq(str(input_dir)).unsqueeze(0).to(device)
    model, lq = _apply_precision(model, lq, precision, device)

    with torch.no_grad(), _autocast_context(device, precision):
        output = test_video(lq, model, rvrt_args)

    _save_output_frames(output=output, output_dir=output_dir, frame_names=[path.name for path in frame_paths])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())