export type OutputMode = "preserveAspect4k" | "cropTo4k" | "native4x";

export type QualityPreset = "qualityMax" | "qualityBalanced" | "vramSafe";

export type ModelId = string;

export type AspectRatioPreset = "source" | "16:9" | "9:16" | "4:3" | "1:1" | "21:9" | "custom";

export type ResolutionBasis = "exact" | "width" | "height";

export type VideoCodec = "h264" | "h265";

export type OutputContainer = "mp4" | "mkv";

export type PytorchRunner = "torch" | "tensorrt";

export type InterpolationMode = "off" | "afterUpscale" | "interpolateOnly";

export type ColorizationMode = "off" | "colorizeOnly" | "beforeUpscale";

export type ColorizationContextKind = "referenceImages";

export type DeepRemasterProcessingMode = "standard" | "high";

export type InterpolationTargetFps = 30 | 60;

export type GpuKind = "discrete" | "integrated" | "unknown";

export interface GpuDevice {
  id: number;
  name: string;
  kind: GpuKind;
}

export interface ExternalResearchRuntimeStatus {
  kind: "external-command";
  commandEnvVar: string;
  configured: boolean;
  source?: "environment" | "repo-default" | "missing";
}

export interface SourceVideoSummary {
  path: string;
  previewPath: string;
  width: number;
  height: number;
  durationSeconds: number;
  frameRate: number;
  hasAudio: boolean;
  container: string;
  videoCodec: string;
  sourceBitrateKbps?: number | null;
  videoProfile?: string | null;
  pixelFormat?: string | null;
  audioCodec?: string | null;
  audioProfile?: string | null;
  audioSampleRate?: number | null;
  audioChannels?: string | null;
  audioBitrateKbps?: number | null;
}

export interface RuntimeStatus {
  ffmpegPath: string;
  realesrganPath: string;
  modelDir: string;
  rifePath?: string;
  rifeModelRoot?: string;
  availableGpus: GpuDevice[];
  defaultGpuId: number | null;
  externalResearchRuntimes?: Record<string, ExternalResearchRuntimeStatus>;
}

export interface ModelRating {
  rating: number;
  updatedAt: string;
}

export interface SourceContextLibraryEntry {
  entryId: string;
  fileName: string;
  relativePath: string;
  absolutePath: string;
  sizeBytes: number;
  createdAt: string;
}

export interface SourceContextLibrary {
  libraryId: string;
  sourcePath: string;
  sourceFileName: string;
  sourceFingerprint: string;
  folderPath: string;
  refsPath: string;
  createdAt: string;
  entries: SourceContextLibraryEntry[];
}

export interface SelectedColorizationContext {
  libraryId: string;
  selectedEntryIds: string[];
  referenceImagePaths: string[];
}

export interface BlindComparisonRecord {
  sourcePath: string;
  previewDurationSeconds: number;
  previewStartOffsetSeconds?: number | null;
  comparisonSignature?: string | null;
  winnerModelId: ModelId;
  candidateModelIds: ModelId[];
  createdAt: string;
}

export interface AppConfig {
  modelRatings: Record<string, ModelRating>;
  blindComparisons: BlindComparisonRecord[];
}

export interface BlindComparisonSelectionInput {
  sourcePath: string;
  previewDurationSeconds: number;
  previewStartOffsetSeconds?: number | null;
  comparisonSignature?: string | null;
  winnerModelId: ModelId;
  candidateModelIds: ModelId[];
}

export interface OutputSizingOptions {
  aspectRatioPreset: AspectRatioPreset;
  customAspectWidth: number | null;
  customAspectHeight: number | null;
  resolutionBasis: ResolutionBasis;
  targetWidth: number | null;
  targetHeight: number | null;
  cropLeft: number | null;
  cropTop: number | null;
  cropWidth: number | null;
  cropHeight: number | null;
}

export interface RealesrganJobRequest extends OutputSizingOptions {
  sourcePath: string;
  modelId: ModelId;
  colorizationMode: ColorizationMode;
  colorizerModelId: ModelId | null;
  colorizationContext: SelectedColorizationContext | null;
  deepremasterProcessingMode: DeepRemasterProcessingMode;
  outputMode: OutputMode;
  qualityPreset: QualityPreset;
  interpolationMode: InterpolationMode;
  interpolationTargetFps: InterpolationTargetFps | null;
  pytorchRunner: PytorchRunner;
  gpuId: number | null;
  previewMode: boolean;
  previewDurationSeconds: number | null;
  previewStartOffsetSeconds?: number | null;
  segmentDurationSeconds: number | null;
  outputPath: string;
  codec: VideoCodec;
  container: OutputContainer;
  tileSize: number;
  fp16: boolean;
  crf: number;
}

export type PipelinePhase = "queued" | "paused" | "extracting" | "colorizing" | "upscaling" | "interpolating" | "encoding" | "remuxing" | "completed" | "failed";

export interface PipelineProgress {
  phase: PipelinePhase;
  percent: number;
  message: string;
  processedFrames: number;
  totalFrames: number;
  extractedFrames: number;
  colorizedFrames: number;
  upscaledFrames: number;
  interpolatedFrames: number;
  encodedFrames: number;
  remuxedFrames: number;
  segmentIndex?: number | null;
  segmentCount?: number | null;
  segmentProcessedFrames?: number | null;
  segmentTotalFrames?: number | null;
  batchIndex?: number | null;
  batchCount?: number | null;
  elapsedSeconds?: number | null;
  averageFramesPerSecond?: number | null;
  rollingFramesPerSecond?: number | null;
  estimatedRemainingSeconds?: number | null;
  processRssBytes?: number | null;
  gpuMemoryUsedBytes?: number | null;
  gpuMemoryTotalBytes?: number | null;
  scratchSizeBytes?: number | null;
  outputSizeBytes?: number | null;
  extractStageSeconds?: number | null;
  colorizeStageSeconds?: number | null;
  upscaleStageSeconds?: number | null;
  interpolateStageSeconds?: number | null;
  encodeStageSeconds?: number | null;
  remuxStageSeconds?: number | null;
}

export interface InterpolationDiagnostics {
  mode: string;
  sourceFps: number;
  outputFps: number;
  sourceFrameCount: number;
  outputFrameCount: number;
  segmentCount: number;
  segmentFrameLimit: number;
  segmentOverlapFrames: number;
}

export interface PipelineMediaSummary {
  width: number;
  height: number;
  frameRate: number;
  durationSeconds: number;
  frameCount: number;
  aspectRatio: number;
  pixelCount: number;
  hasAudio?: boolean | null;
  container?: string | null;
  videoCodec?: string | null;
}

export interface PipelineEffectiveSettings {
  backendId?: string | null;
  qualityPreset?: string | null;
  requestedTileSize?: number | null;
  effectiveTileSize: number;
  requestedPrecision?: string | null;
  selectedPrecision?: string | null;
  effectivePrecision?: string | null;
  precisionSource?: string | null;
  processedDurationSeconds: number;
  previewStartOffsetSeconds?: number | null;
  segmentFrameLimit: number;
  previewMode: boolean;
  previewDurationSeconds?: number | null;
  segmentDurationSeconds?: number | null;
}

export interface PipelineStageTimings {
  extractSeconds: number;
  colorizeSeconds: number;
  upscaleSeconds: number;
  interpolateSeconds: number;
  encodeSeconds: number;
  remuxSeconds: number;
}

export interface PipelineResourcePeaks {
  processRssBytes?: number | null;
  gpuMemoryUsedBytes?: number | null;
  gpuMemoryTotalBytes?: number | null;
  scratchSizeBytes?: number | null;
  outputSizeBytes?: number | null;
}

export interface ManagedPipelineRunDetails {
  request: RealesrganJobRequest;
  sourceMedia?: PipelineMediaSummary | null;
  outputMedia?: PipelineMediaSummary | null;
  effectiveSettings?: PipelineEffectiveSettings | null;
  executionPath?: string | null;
  videoEncoder?: string | null;
  videoEncoderLabel?: string | null;
  runner?: string | null;
  precision?: string | null;
  torchCompileEnabled?: boolean | null;
  torchCompileMode?: string | null;
  torchCompileCudagraphs?: boolean | null;
  stageTimings?: PipelineStageTimings | null;
  resourcePeaks?: PipelineResourcePeaks | null;
  modelRuntime?: Record<string, unknown> | null;
  averageThroughputFps?: number | null;
  segmentCount?: number | null;
  segmentFrameLimit?: number | null;
  frameCount?: number | null;
  hadAudio?: boolean | null;
  runtime?: RuntimeStatus | null;
}

export interface RealesrganJobPlan {
  model: string;
  command: string[];
  cacheKey: string;
  notes: string[];
}

export interface PipelineResult {
  outputPath: string;
  workDir: string;
  frameCount: number;
  hadAudio: boolean;
  codec: VideoCodec;
  container: OutputContainer;
  sourceMedia?: PipelineMediaSummary | null;
  outputMedia?: PipelineMediaSummary | null;
  effectiveSettings?: PipelineEffectiveSettings | null;
  interpolationDiagnostics?: InterpolationDiagnostics | null;
  executionPath?: string | null;
  videoEncoder?: string | null;
  videoEncoderLabel?: string | null;
  runner?: string | null;
  precision?: string | null;
  torchCompileEnabled?: boolean | null;
  torchCompileMode?: string | null;
  torchCompileCudagraphs?: boolean | null;
  runtime: RuntimeStatus;
  stageTimings?: PipelineStageTimings | null;
  resourcePeaks?: PipelineResourcePeaks | null;
  modelRuntime?: Record<string, unknown> | null;
  averageThroughputFps?: number | null;
  segmentCount?: number | null;
  segmentFrameLimit?: number | null;
  log: string[];
}

export interface PathStats {
  path: string;
  exists: boolean;
  isDirectory: boolean;
  sizeBytes: number;
}

export interface ScratchStorageSummary {
  jobsRoot: PathStats;
  convertedSourcesRoot: PathStats;
  sourcePreviewsRoot: PathStats;
}

export interface ManagedJobSummary {
  jobId: string;
  jobKind: "pipeline" | "sourceConversion" | string;
  label: string;
  state: "queued" | "running" | "paused" | "succeeded" | "failed" | "cancelled" | "interrupted" | string;
  sourcePath: string | null;
  modelId: string | null;
  codec: string | null;
  container: string | null;
  progress: PipelineProgress;
  recordedCount: number;
  scratchPath: string | null;
  scratchStats: PathStats | null;
  outputPath: string | null;
  outputStats: PathStats | null;
  pipelineRunDetails?: ManagedPipelineRunDetails | null;
  updatedAt: string;
}

export interface PipelineJobStatus {
  jobId: string;
  state: "queued" | "running" | "paused" | "succeeded" | "failed" | "cancelled";
  progress: PipelineProgress;
  result: PipelineResult | null;
  error: string | null;
}

export interface SourceConversionJobStatus {
  jobId: string;
  state: "queued" | "running" | "paused" | "succeeded" | "failed" | "cancelled";
  progress: PipelineProgress;
  result: SourceVideoSummary | null;
  error: string | null;
}

export interface DesktopApi {
  selectVideoFile(): Promise<string | null>;
  selectContextFiles(): Promise<string[]>;
  selectOutputFile(defaultPath: string, container: OutputContainer): Promise<string | null>;
  ensureRuntimeAssets(): Promise<RuntimeStatus>;
  probeSourceVideo(sourcePath: string): Promise<SourceVideoSummary>;
  getSourceContextLibrary(sourcePath: string): Promise<SourceContextLibrary | null>;
  importSourceContextFiles(sourcePath: string, importPaths: string[]): Promise<SourceContextLibrary>;
  removeSourceContextEntry(sourcePath: string, entryId: string): Promise<SourceContextLibrary>;
  startSourceConversionToMp4(sourcePath: string): Promise<string>;
  getSourceConversionJob(jobId: string): Promise<SourceConversionJobStatus>;
  pauseSourceConversionJob(jobId: string): Promise<void>;
  resumeSourceConversionJob(jobId: string): Promise<void>;
  cancelSourceConversionJob(jobId: string): Promise<void>;
  getAppConfig(): Promise<AppConfig>;
  saveModelRating(modelId: ModelId, rating: number | null): Promise<AppConfig>;
  recordBlindComparisonSelection(selection: BlindComparisonSelectionInput): Promise<AppConfig>;
  startPipeline(request: RealesrganJobRequest): Promise<string>;
  getPipelineJob(jobId: string): Promise<PipelineJobStatus>;
  pausePipelineJob(jobId: string): Promise<void>;
  resumePipelineJob(jobId: string): Promise<void>;
  cancelPipelineJob(jobId: string): Promise<void>;
  getPathStats(path: string): Promise<PathStats>;
  getScratchStorageSummary(): Promise<ScratchStorageSummary>;
  listManagedJobs(): Promise<ManagedJobSummary[]>;
  deleteManagedPath(path: string): Promise<void>;
  openPathInDefaultApp(path: string): Promise<void>;
  loadPreviewUrl(path: string): Promise<string>;
  toPreviewSrc(path: string): string;
}
