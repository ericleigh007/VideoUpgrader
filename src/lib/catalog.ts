import catalogData from "../../config/model_catalog.json";
import type { ModelId } from "../types";

export interface BackendDefinition {
  id: string;
  label: string;
  loader: string;
  runtime: string;
  supportsGpuSelection: boolean;
  frameTransport: string;
}

export interface ModelHandlingDefinition {
  cropMath: string;
  requiresVulkan: boolean;
  supportsGpuId: boolean;
  prefersFixedScaleModelOutput: boolean;
  variantSuffixByScale?: boolean;
}

export interface ModelRuntimeAssetDefinition {
  kind: "checkpoint";
  fileName: string;
  downloadUrl: string;
}

export interface ModelResearchRuntimeDefinition {
  kind: "external-command";
  commandEnvVar: string;
}

export type ModelExecutionStatus = "runnable" | "planned";

export type ModelSupportTier = "production" | "next" | "research" | "compatibility";

export interface ModelDefinition {
  value: ModelId;
  backendId: string;
  loader: string;
  runtimeModelName: string;
  label: string;
  summary: string;
  visibleInUi: boolean;
  comparisonEligible: boolean;
  executionStatus: ModelExecutionStatus;
  supportTier: ModelSupportTier;
  qualityRank: number;
  videoNative: boolean;
  mediaSuitability: string[];
  nativeScale: number;
  runtimeAsset?: ModelRuntimeAssetDefinition;
  researchRuntime?: ModelResearchRuntimeDefinition;
  specialHandling: ModelHandlingDefinition;
}

interface CatalogData {
  backends: BackendDefinition[];
  models: Array<{
    id: string;
    backendId: string;
    loader: string;
    runtimeModelName: string;
    label: string;
    summary: string;
    visibleInUi: boolean;
    comparisonEligible: boolean;
    executionStatus: ModelExecutionStatus;
    supportTier: ModelSupportTier;
    qualityRank: number;
    videoNative: boolean;
    mediaSuitability: string[];
    nativeScale: number;
    runtimeAsset?: ModelRuntimeAssetDefinition;
    researchRuntime?: ModelResearchRuntimeDefinition;
    specialHandling: ModelHandlingDefinition;
  }>;
}

const typedCatalogData = catalogData as CatalogData;

export const backendCatalog: BackendDefinition[] = typedCatalogData.backends;

export const modelCatalog: ModelDefinition[] = typedCatalogData.models.map((model) => ({
  value: model.id,
  backendId: model.backendId,
  loader: model.loader,
  runtimeModelName: model.runtimeModelName,
  label: model.label,
  summary: model.summary,
  visibleInUi: model.visibleInUi,
  comparisonEligible: model.comparisonEligible,
  executionStatus: model.executionStatus,
  supportTier: model.supportTier,
  qualityRank: model.qualityRank,
  videoNative: model.videoNative,
  mediaSuitability: model.mediaSuitability,
  nativeScale: model.nativeScale,
  runtimeAsset: model.runtimeAsset,
  researchRuntime: model.researchRuntime,
  specialHandling: model.specialHandling,
}));

export function getUiModels(): ModelDefinition[] {
  return modelCatalog
    .filter((model) => model.visibleInUi || model.executionStatus === "planned")
    .sort((left, right) => {
      if (left.qualityRank !== right.qualityRank) {
        return left.qualityRank - right.qualityRank;
      }
      return left.label.localeCompare(right.label);
    });
}

export function getVisibleModels(): ModelDefinition[] {
  return modelCatalog.filter((model) => model.visibleInUi);
}

export function getBlindComparisonModels(): ModelDefinition[] {
  return modelCatalog.filter((model) => model.comparisonEligible && model.executionStatus === "runnable");
}

export function getTopRatedModels(): ModelDefinition[] {
  return modelCatalog
    .filter((model) => model.qualityRank <= 6)
    .sort((left, right) => left.qualityRank - right.qualityRank);
}

export function getModelDefinition(modelId: ModelId): ModelDefinition {
  const model = modelCatalog.find((entry) => entry.value === modelId);
  if (!model) {
    throw new Error(`Unknown model: ${modelId}`);
  }

  return model;
}

export function getBackendDefinition(backendId: string): BackendDefinition {
  const backend = backendCatalog.find((entry) => entry.id === backendId);
  if (!backend) {
    throw new Error(`Unknown backend: ${backendId}`);
  }

  return backend;
}