from __future__ import annotations

SUPPORTED_PRECISION_MODES = {"fp32", "fp16", "bf16"}


def resolve_precision_mode(*, fp16: bool = False, bf16: bool = False, precision: str | None = None) -> str:
    if precision is not None:
        resolved = precision.strip().lower()
        if resolved not in SUPPORTED_PRECISION_MODES:
            supported = ", ".join(sorted(SUPPORTED_PRECISION_MODES))
            raise ValueError(f"Unsupported precision mode '{precision}'. Expected one of: {supported}")
        if fp16 and resolved != "fp16":
            raise ValueError("precision conflicts with fp16 flag")
        if bf16 and resolved != "bf16":
            raise ValueError("precision conflicts with bf16 flag")
        if fp16 and bf16:
            raise ValueError("fp16 and bf16 cannot be requested at the same time")
        return resolved

    if fp16 and bf16:
        raise ValueError("fp16 and bf16 cannot be requested at the same time")
    if fp16:
        return "fp16"
    if bf16:
        return "bf16"
    return "fp32"