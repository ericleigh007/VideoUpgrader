from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import tensorrt as trt
import torch

from upscaler_worker.runtime import runtime_root


TENSORRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TENSORRT_INPUT_NAME = "input"
TENSORRT_OUTPUT_NAME = "output"
TENSORRT_ONNX_OPSET = 17


class TensorRtImageModelRunner:
    def __init__(
        self,
        *,
        model_id: str,
        checkpoint_path: Path,
        torch_model: torch.nn.Module,
        device: torch.device,
        scale: int,
        precision_mode: str,
        log: list[str],
    ) -> None:
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.scale = scale
        self.precision_mode = precision_mode
        self._torch_model = torch_model.eval().cpu()
        self._log = log
        self._cache_dir = runtime_root() / "tensorrt-models" / model_id / precision_mode
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._engine_cache: dict[tuple[int, int, int, int], tuple[trt.ICudaEngine, trt.IExecutionContext]] = {}
        self._engine_lock = threading.Lock()
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._log.append("Using a dedicated CUDA stream for TensorRT execution.")

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.device.type != "cuda":
            raise RuntimeError("TensorRT runner requires CUDA input tensors")
        if self._stream is None:
            raise RuntimeError("TensorRT runner CUDA stream is unavailable")

        shape = tuple(int(value) for value in image_tensor.shape)
        engine, context = self._get_or_create_engine(shape)
        context.set_input_shape(TENSORRT_INPUT_NAME, shape)
        current_stream = torch.cuda.current_stream(image_tensor.device)
        self._stream.wait_stream(current_stream)

        with torch.cuda.stream(self._stream):
            input_tensor = image_tensor.contiguous()
            output_shape = tuple(int(value) for value in context.get_tensor_shape(TENSORRT_OUTPUT_NAME))
            output_tensor = torch.empty(output_shape, device=image_tensor.device, dtype=torch.float32)
            context.set_tensor_address(TENSORRT_INPUT_NAME, input_tensor.data_ptr())
            context.set_tensor_address(TENSORRT_OUTPUT_NAME, output_tensor.data_ptr())
            if not context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed for shape {shape}")
            input_tensor.record_stream(self._stream)
            output_tensor.record_stream(self._stream)

        current_stream.wait_stream(self._stream)
        output_tensor.record_stream(current_stream)
        return output_tensor

    def _get_or_create_engine(self, shape: tuple[int, int, int, int]) -> tuple[trt.ICudaEngine, trt.IExecutionContext]:
        with self._engine_lock:
            cached = self._engine_cache.get(shape)
            if cached is not None:
                return cached

            shape_tag = "x".join(str(value) for value in shape)
            shape_hash = hashlib.sha256(shape_tag.encode("utf-8")).hexdigest()[:12]
            onnx_path = self._cache_dir / f"{shape_tag}-{shape_hash}.onnx"
            engine_path = self._cache_dir / f"{shape_tag}-{shape_hash}.engine"

            if not onnx_path.exists():
                self._export_onnx(shape, onnx_path)
            if not engine_path.exists():
                self._build_engine(onnx_path, engine_path, shape)

            engine = self._load_engine(engine_path)
            context = engine.create_execution_context()
            created = (engine, context)
            self._engine_cache[shape] = created
            return created

    def _export_onnx(self, shape: tuple[int, int, int, int], onnx_path: Path) -> None:
        dummy_input = torch.rand(shape, dtype=torch.float32)
        with torch.inference_mode():
            torch.onnx.export(
                self._torch_model,
                dummy_input,
                str(onnx_path),
                input_names=[TENSORRT_INPUT_NAME],
                output_names=[TENSORRT_OUTPUT_NAME],
                opset_version=TENSORRT_ONNX_OPSET,
                dynamo=False,
            )
        self._log.append(f"Exported TensorRT ONNX graph for shape {shape} to {onnx_path.name}.")

    def _build_engine(self, onnx_path: Path, engine_path: Path, shape: tuple[int, int, int, int]) -> None:
        builder = trt.Builder(TENSORRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TENSORRT_LOGGER)
        if not parser.parse_from_file(str(onnx_path)):
            errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"TensorRT failed to parse {onnx_path.name}: {errors}")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.TF32)
        resolved_precision = self._configure_precision_flags(builder, config)
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"TensorRT failed to build an engine for shape {shape}")

        engine_path.write_bytes(bytes(serialized_engine))
        self._log.append(f"Built TensorRT engine for shape {shape} ({resolved_precision}) at {engine_path.name}.")

    def _configure_precision_flags(self, builder: trt.Builder, config: trt.IBuilderConfig) -> str:
        if self.precision_mode == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                return "fp16"
            self._log.append("TensorRT fp16 requested, but the platform did not report fast fp16 support. Using fp32.")
            return "fp32"

        if self.precision_mode == "bf16":
            if hasattr(trt.BuilderFlag, "BF16"):
                config.set_flag(trt.BuilderFlag.BF16)
                return "bf16"
            self._log.append("TensorRT bf16 requested, but this runtime does not expose BF16 builder support. Using fp32.")
            return "fp32"

        return "fp32"

    def _load_engine(self, engine_path: Path) -> trt.ICudaEngine:
        runtime = trt.Runtime(TENSORRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize engine at {engine_path}")
        return engine