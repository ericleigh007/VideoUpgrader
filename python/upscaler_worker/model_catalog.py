from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_model_catalog() -> dict[str, object]:
    catalog_path = repo_root() / "config" / "model_catalog.json"
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def backend_catalog() -> list[dict[str, object]]:
    catalog = load_model_catalog()
    return list(catalog.get("backends", []))


def model_catalog() -> list[dict[str, object]]:
    catalog = load_model_catalog()
    return list(catalog.get("models", []))


def get_model_definition(model_id: str) -> dict[str, object]:
    for model in model_catalog():
        if model.get("id") == model_id:
            return model

    supported = ", ".join(sorted(str(model.get("id")) for model in model_catalog()))
    raise ValueError(f"Unsupported model '{model_id}'. Supported models: {supported}")


def get_backend_definition(backend_id: str) -> dict[str, object]:
    for backend in backend_catalog():
        if backend.get("id") == backend_id:
            return backend

    raise ValueError(f"Unknown backend '{backend_id}'")


def ensure_supported_model(model_id: str) -> str:
    get_model_definition(model_id)
    return model_id


def model_label(model_id: str) -> str:
    return str(get_model_definition(model_id)["label"])


def model_backend_id(model_id: str) -> str:
    return str(get_model_definition(model_id)["backendId"])


def model_loader(model_id: str) -> str:
    return str(get_model_definition(model_id)["loader"])


def model_runtime_name(model_id: str) -> str:
    return str(get_model_definition(model_id)["runtimeModelName"])


def model_native_scale(model_id: str) -> int:
    return int(get_model_definition(model_id)["nativeScale"])


def model_runtime_asset(model_id: str) -> dict[str, object] | None:
    asset = get_model_definition(model_id).get("runtimeAsset")
    if not isinstance(asset, dict):
        return None
    return dict(asset)


def model_research_runtime(model_id: str) -> dict[str, object] | None:
    runtime = get_model_definition(model_id).get("researchRuntime")
    if not isinstance(runtime, dict):
        return None
    return dict(runtime)


def model_special_handling(model_id: str) -> dict[str, object]:
    return dict(get_model_definition(model_id).get("specialHandling", {}))


def model_support_tier(model_id: str) -> str:
    return str(get_model_definition(model_id).get("supportTier", "next"))


def visible_ui_models() -> list[dict[str, object]]:
    return [model for model in model_catalog() if bool(model.get("visibleInUi"))]


def comparison_eligible_models() -> list[dict[str, object]]:
    return [
        model for model in model_catalog()
        if bool(model.get("comparisonEligible")) and str(model.get("executionStatus")) == "runnable"
    ]


def top_rated_models() -> list[dict[str, object]]:
    ranked_models = [model for model in model_catalog() if int(model.get("qualityRank", 999)) <= 6]
    return sorted(ranked_models, key=lambda model: int(model.get("qualityRank", 999)))


def model_execution_status(model_id: str) -> str:
    return str(get_model_definition(model_id).get("executionStatus", "planned"))


def ensure_runnable_model(model_id: str) -> str:
    ensure_supported_model(model_id)
    if model_execution_status(model_id) != "runnable":
        raise ValueError(f"Model '{model_id}' is cataloged but not yet runnable in the current app build")
    return model_id


def ensure_benchmarkable_model(model_id: str) -> str:
    ensure_supported_model(model_id)
    if model_execution_status(model_id) == "runnable":
        return model_id

    if model_backend_id(model_id) == "pytorch-video-sr" and model_support_tier(model_id) == "research":
        return model_id

    raise ValueError(f"Model '{model_id}' is not benchmarkable in the current app build")