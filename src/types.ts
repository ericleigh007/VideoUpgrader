export type OutputMode = "preserveAspect4k" | "cropTo4k" | "native4x";

export type QualityPreset = "qualityMax" | "qualityBalanced" | "vramSafe";

export type ModelId = string;

export type AspectRatioPreset = "source" | "16:9" | "9:16" | "4:3" | "1:1" | "21:9" | "custom";

export type ResolutionBasis = "exact" | "width" | "height";

export type VideoCodec = "h264" | "h265";

export type OutputContainer = "mp4" | "mkv";

export type GpuKind = "discrete" | "integrated" | "unknown";

export interface GpuDevice {
  id: number;
  name: string;
  kind: GpuKind;
}

export interface SourceVideoSummary {
  path: string;
  width: number;
  height: number;
  durationSeconds: number;
  frameRate: number;
  hasAudio: boolean;
  container: string;
}

export interface RuntimeStatus {
  ffmpegPath: string;
  realesrganPath: string;
  modelDir: string;
  availableGpus: GpuDevice[];
  defaultGpuId: number | null;
}

export interface ModelRating {
  rating: number;
  updatedAt: string;
}

export interface BlindComparisonRecord {
  sourcePath: string;
  previewDurationSeconds: number;
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
  outputMode: OutputMode;
  qualityPreset: QualityPreset;
  gpuId: number | null;
  previewMode: boolean;
  previewDurationSeconds: number | null;
  outputPath: string;
  codec: VideoCodec;
  container: OutputContainer;
  tileSize: number;
  fp16: boolean;
  crf: number;
}

export type PipelinePhase = "queued" | "extracting" | "upscaling" | "encoding" | "remuxing" | "completed" | "failed";

export interface PipelineProgress {
  phase: PipelinePhase;
  percent: number;
  message: string;
  processedFrames: number;
  totalFrames: number;
  extractedFrames: number;
  upscaledFrames: number;
  encodedFrames: number;
  remuxedFrames: number;
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
  runtime: RuntimeStatus;
  log: string[];
}

export interface PipelineJobStatus {
  jobId: string;
  state: "queued" | "running" | "succeeded" | "failed";
  progress: PipelineProgress;
  result: PipelineResult | null;
  error: string | null;
}

export interface DesktopApi {
  selectVideoFile(): Promise<string | null>;
  selectOutputFile(defaultPath: string, container: OutputContainer): Promise<string | null>;
  ensureRuntimeAssets(): Promise<RuntimeStatus>;
  probeSourceVideo(sourcePath: string): Promise<SourceVideoSummary>;
  getAppConfig(): Promise<AppConfig>;
  saveModelRating(modelId: ModelId, rating: number | null): Promise<AppConfig>;
  recordBlindComparisonSelection(selection: BlindComparisonSelectionInput): Promise<AppConfig>;
  startPipeline(request: RealesrganJobRequest): Promise<string>;
  getPipelineJob(jobId: string): Promise<PipelineJobStatus>;
  openPathInDefaultApp(path: string): Promise<void>;
  toPreviewSrc(path: string): string;
}
