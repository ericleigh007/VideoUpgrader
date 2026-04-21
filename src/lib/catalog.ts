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

export type ModelContextKind = "referenceImages";

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
  task: "upscale" | "colorize";
  visibleInUi: boolean;
  comparisonEligible: boolean;
  executionStatus: ModelExecutionStatus;
  supportTier: ModelSupportTier;
  qualityRank: number;
  supportsContextInput?: boolean;
  supportedContextKinds?: ModelContextKind[];
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
    task: "upscale" | "colorize";
    visibleInUi: boolean;
    comparisonEligible: boolean;
    executionStatus: ModelExecutionStatus;
    supportTier: ModelSupportTier;
    qualityRank: number;
    supportsContextInput?: boolean;
    supportedContextKinds?: ModelContextKind[];
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
  task: model.task,
  visibleInUi: model.visibleInUi,
  comparisonEligible: model.comparisonEligible,
  executionStatus: model.executionStatus,
  supportTier: model.supportTier,
  qualityRank: model.qualityRank,
  supportsContextInput: model.supportsContextInput ?? false,
  supportedContextKinds: model.supportedContextKinds ?? [],
  videoNative: model.videoNative,
  mediaSuitability: model.mediaSuitability,
  nativeScale: model.nativeScale,
  runtimeAsset: model.runtimeAsset,
  researchRuntime: model.researchRuntime,
  specialHandling: model.specialHandling,
}));

export function getUiModels(): ModelDefinition[] {
  return modelCatalog
    .filter((model) => model.task === "upscale" && (model.visibleInUi || model.executionStatus === "planned"))
    .sort((left, right) => {
      if (left.qualityRank !== right.qualityRank) {
        return left.qualityRank - right.qualityRank;
      }
      return left.label.localeCompare(right.label);
    });
}

export function getVisibleModels(): ModelDefinition[] {
  return modelCatalog.filter((model) => model.task === "upscale" && model.visibleInUi);
}

export function getBlindComparisonModels(): ModelDefinition[] {
  return modelCatalog.filter((model) => model.task === "upscale" && model.comparisonEligible && model.executionStatus === "runnable");
}

export function getUiColorizerModels(): ModelDefinition[] {
  return modelCatalog
    .filter((model) => model.task === "colorize" && (model.visibleInUi || model.executionStatus === "planned"))
    .sort((left, right) => {
      if (left.qualityRank !== right.qualityRank) {
        return left.qualityRank - right.qualityRank;
      }
      return left.label.localeCompare(right.label);
    });
}

export function getVisibleColorizerModels(): ModelDefinition[] {
  return modelCatalog.filter((model) => model.task === "colorize" && model.visibleInUi);
}

export function getBlindComparisonColorizerModels(): ModelDefinition[] {
  return modelCatalog.filter((model) => model.task === "colorize" && model.comparisonEligible && model.executionStatus === "runnable");
}

export function getTopRatedModels(): ModelDefinition[] {
  return modelCatalog
    .filter((model) => model.task === "upscale" && model.qualityRank <= 6)
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