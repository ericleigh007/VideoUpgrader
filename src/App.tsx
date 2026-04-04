import { Fragment, useEffect, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from "react";
import { desktopApi } from "./lib/desktopApi";
import { getBackendDefinition, getBlindComparisonModels, getModelDefinition, getUiModels } from "./lib/catalog";
import { defaultCropRect, planOutputFraming, resolveAspectRatio, resolveCropRect, type NormalizedCropRect } from "./lib/framing";
import type {
  AppConfig,
  AspectRatioPreset,
  ManagedJobSummary,
  ModelId,
  OutputContainer,
  OutputMode,
  OutputSizingOptions,
  PathStats,
  PipelineProgress,
  PipelineJobStatus,
  PipelineResult,
  PytorchRunner,
  QualityPreset,
  RealesrganJobRequest,
  ResolutionBasis,
  RuntimeStatus,
  ScratchStorageSummary,
  SourceConversionJobStatus,
  SourceVideoSummary,
  VideoCodec,
} from "./types";

const models = getUiModels();
const blindComparisonCandidates = getBlindComparisonModels();
const runnableModels = models.filter((model) => model.executionStatus === "runnable");
const plannedModels = models.filter((model) => model.executionStatus !== "runnable");

const outputModes: Array<{ value: OutputMode; label: string }> = [
  { value: "preserveAspect4k", label: "Preserve Aspect In Target" },
  { value: "cropTo4k", label: "Crop To Fill Target" },
  { value: "native4x", label: "Native 2x (4x Pixels)" }
];

const qualityPresets: Array<{ value: QualityPreset; label: string }> = [
  { value: "qualityMax", label: "Quality Max" },
  { value: "qualityBalanced", label: "Quality Balanced" },
  { value: "vramSafe", label: "VRAM Safe" }
];

const codecs: Array<{ value: VideoCodec; label: string }> = [
  { value: "h264", label: "H.264" },
  { value: "h265", label: "H.265" }
];

const containers: Array<{ value: OutputContainer; label: string }> = [
  { value: "mp4", label: "MP4" },
  { value: "mkv", label: "MKV" }
];

const pytorchRunners: Array<{ value: PytorchRunner; label: string }> = [
  { value: "torch", label: "Torch" },
  { value: "tensorrt", label: "TensorRT" }
];

function recommendedPytorchRunner(modelId: ModelId): PytorchRunner {
  const model = getModelDefinition(modelId);
  if (model.backendId === "pytorch-image-sr" && model.value === "swinir-realworld-x4") {
    return "tensorrt";
  }
  return "torch";
}

const aspectRatioPresets: Array<{ value: AspectRatioPreset; label: string }> = [
  { value: "source", label: "Match Source" },
  { value: "16:9", label: "16:9 Landscape" },
  { value: "9:16", label: "9:16 Vertical" },
  { value: "4:3", label: "4:3 Classic" },
  { value: "1:1", label: "1:1 Square" },
  { value: "21:9", label: "21:9 Cinematic" },
  { value: "custom", label: "Custom Ratio" }
];

const resolutionBases: Array<{ value: ResolutionBasis; label: string }> = [
  { value: "exact", label: "Exact Width x Height" },
  { value: "width", label: "Width Driven" },
  { value: "height", label: "Height Driven" }
];

type CropHandle = "move" | "nw" | "ne" | "sw" | "se";

interface DragState {
  handle: CropHandle;
  startX: number;
  startY: number;
  startRect: NormalizedCropRect;
}

interface BlindComparisonEntry {
  sampleId: string;
  anonymousLabel: string;
  modelId: ModelId;
  jobId: string | null;
  status: PipelineJobStatus;
}

interface BlindComparisonState {
  state: "running" | "ready" | "failed";
  entries: BlindComparisonEntry[];
  previewDurationSeconds: number;
  selectedSampleId: string | null;
  winnerModelId: ModelId | null;
  revealed: boolean;
  error: string | null;
}

interface ComparisonFocusPreset {
  id: string;
  label: string;
  focusX: number;
  focusY: number;
  hint: string;
}

interface ExpandablePanelProps {
  title: string;
  subtitle?: string;
  isOpen: boolean;
  onToggle: () => void;
  testId: string;
  children: ReactNode;
}

interface TrackedJobEntry {
  id: string;
  jobKind: string;
  label: string;
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  phase: string;
  progress: PipelineProgress;
  modelId: string | null;
  codec: string | null;
  container: string | null;
  recordedCount: number;
  message: string;
  updatedAt: number;
  sourcePath: string | null;
  scratchPath: string | null;
  scratchSizeBytes: number;
  outputPath: string | null;
  outputSizeBytes: number;
  onStop: null | (() => void);
  onClearScratch: null | (() => void);
  onDeleteOutput: null | (() => void);
}

interface ProgressEventEntry {
  key: string;
  title: string;
  detail: string;
  percent: number;
  timestamp: number;
}

type CleanupJobFilter = "all" | "running" | "succeeded" | "cancelled" | "failed";
type CleanupJobSort = "largest" | "newest" | "oldest";

const CLEANUP_FILTER_STORAGE_KEY = "upscaler.cleanup.filter";
const CLEANUP_SEARCH_STORAGE_KEY = "upscaler.cleanup.search";
const CLEANUP_SORT_STORAGE_KEY = "upscaler.cleanup.sort";
const EMBEDDED_PREVIEW_COMPATIBLE_CONTAINERS = new Set(["mp4"]);
const AUTO_PREVIEW_UPGRADE_MAX_DURATION_SECONDS = 300;

const comparisonFocusPresets: ComparisonFocusPreset[] = [
  {
    id: "dithering",
    label: "Dithering And Macroblocks",
    focusX: 50,
    focusY: 50,
    hint: "Check flat gradients, skin tones, and compressed shadows to see whether dithering turns into chunky pixel patterns or smoother tonal transitions.",
  },
  {
    id: "diagonals",
    label: "Diagonal Anti-Aliasing",
    focusX: 64,
    focusY: 36,
    hint: "Inspect slanted edges like railings, roof lines, and text strokes to see whether stair-stepping becomes cleaner diagonal detail or a sharpened halo.",
  },
  {
    id: "horizontal",
    label: "Horizontal Detail",
    focusX: 50,
    focusY: 24,
    hint: "Look at long horizontal contours such as horizons, shelves, and eyelids to compare line stability, ringing, and oversharpened edges.",
  },
  {
    id: "vertical",
    label: "Vertical Detail",
    focusX: 26,
    focusY: 50,
    hint: "Look at door frames, poles, and facial outlines to judge whether vertical features stay crisp or become warped, doubled, or noisy.",
  },
];

function ExpandablePanel({ title, subtitle, isOpen, onToggle, testId, children }: ExpandablePanelProps) {
  return (
    <article className={`panel expandable-panel${isOpen ? " expandable-panel-open" : ""}`} data-testid={testId}>
      <button type="button" className="expandable-panel-header" onClick={onToggle} aria-expanded={isOpen} data-testid={testId ? `${testId}-toggle` : undefined}>
        <div className="expandable-panel-title-block">
          <h2>{title}</h2>
          {subtitle ? <span className="expandable-panel-subtitle">{subtitle}</span> : null}
        </div>
        <span className="expandable-panel-indicator">{isOpen ? "Hide" : "Show"}</span>
      </button>
      {isOpen ? <div className="expandable-panel-content">{children}</div> : null}
    </article>
  );
}

function normalizeOutputPath(path: string, container: OutputContainer): string {
  const suffix = `.${container}`;
  return path.toLowerCase().endsWith(suffix) ? path : `${path}${suffix}`;
}

function defaultOutputPath(source: SourceVideoSummary | null, container: OutputContainer, modelId: ModelId): string {
  const stem = source?.path.replace(/\\/g, "/").split("/").pop()?.replace(/\.[^.]+$/, "") ?? "upscaled_output";
  const modelStem = modelId.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  return `artifacts/outputs/${stem}_${modelStem}.${container}`;
}

function pathLeaf(path: string | null | undefined): string {
  if (!path) {
    return "n/a";
  }
  const normalized = path.replace(/\\/g, "/");
  const candidate = normalized.split("/").pop();
  return candidate && candidate.length > 0 ? candidate : normalized;
}

function jobDirectoryLabel(path: string | null | undefined): string {
  if (!path) {
    return "n/a";
  }
  return pathLeaf(path);
}

function blindComparisonOutputPath(
  source: SourceVideoSummary,
  container: OutputContainer,
  modelId: ModelId,
  anonymousLabel: string,
  runToken: string,
): string {
  const stem = source.path.replace(/\\/g, "/").split("/").pop()?.replace(/\.[^.]+$/, "") ?? "comparison_source";
  const modelStem = modelId.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  const labelStem = anonymousLabel.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  return `artifacts/outputs/blind/${stem}_${runToken}_${labelStem}_${modelStem}.${container}`;
}

function parsePositiveIntegerInput(value: string): number | null {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function supportsEmbeddedFullLengthPreview(container: string | null | undefined): boolean {
  return EMBEDDED_PREVIEW_COMPATIBLE_CONTAINERS.has(String(container ?? "").toLowerCase());
}

function previewMimeType(path: string | null | undefined): string | undefined {
  const normalized = String(path ?? '').toLowerCase();
  if (normalized.endsWith('.mp4')) {
    return 'video/mp4';
  }
  if (normalized.endsWith('.webm')) {
    return 'video/webm';
  }
  if (normalized.endsWith('.mov')) {
    return 'video/quicktime';
  }
  if (normalized.endsWith('.mkv')) {
    return 'video/x-matroska';
  }
  return undefined;
}

function clampNormalized(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function clampCropRect(rect: NormalizedCropRect): NormalizedCropRect {
  const width = clampNormalized(rect.width);
  const height = clampNormalized(rect.height);
  return {
    width,
    height,
    left: Math.min(Math.max(0, rect.left), Math.max(0, 1 - width)),
    top: Math.min(Math.max(0, rect.top), Math.max(0, 1 - height))
  };
}

function offsetCropRect(rect: NormalizedCropRect, deltaLeft: number, deltaTop: number): NormalizedCropRect {
  return clampCropRect({
    ...rect,
    left: rect.left + deltaLeft,
    top: rect.top + deltaTop,
  });
}

function resizeCropRect(
  startRect: NormalizedCropRect,
  handle: Exclude<CropHandle, "move">,
  deltaX: number,
  deltaY: number,
  aspectRatio: number,
  sourceAspectRatio: number,
): NormalizedCropRect {
  const normalizedAspectRatio = sourceAspectRatio > 0 ? aspectRatio / sourceAspectRatio : aspectRatio;
  const leftAnchor = handle === "ne" || handle === "se" ? startRect.left : startRect.left + startRect.width;
  const topAnchor = handle === "sw" || handle === "se" ? startRect.top : startRect.top + startRect.height;
  const signedWidth = (handle === "ne" || handle === "se" ? 1 : -1) * deltaX;
  const signedHeight = (handle === "sw" || handle === "se" ? 1 : -1) * deltaY;
  const nextWidth = Math.max(0.08, startRect.width + signedWidth + (signedHeight * normalizedAspectRatio));
  const constrainedWidth = Math.min(nextWidth, 1);
  const constrainedHeight = normalizedAspectRatio > 0 ? constrainedWidth / normalizedAspectRatio : constrainedWidth;
  const nextLeft = handle === "ne" || handle === "se" ? leftAnchor : leftAnchor - constrainedWidth;
  const nextTop = handle === "sw" || handle === "se" ? topAnchor : topAnchor - constrainedHeight;
  return clampCropRect({ left: nextLeft, top: nextTop, width: constrainedWidth, height: constrainedHeight });
}

function createQueuedJob(jobId: string): PipelineJobStatus {
  return {
    jobId,
    state: "queued",
    progress: {
      phase: "queued",
      percent: 0,
      message: "Job queued",
      processedFrames: 0,
      totalFrames: 0,
      extractedFrames: 0,
      upscaledFrames: 0,
      encodedFrames: 0,
      remuxedFrames: 0,
    },
    result: null,
    error: null,
  };
}

function createPendingComparisonJob(): PipelineJobStatus {
  return {
    jobId: "pending",
    state: "queued",
    progress: {
      phase: "queued",
      percent: 0,
      message: "Waiting to start",
      processedFrames: 0,
      totalFrames: 0,
      extractedFrames: 0,
      upscaledFrames: 0,
      encodedFrames: 0,
      remuxedFrames: 0,
    },
    result: null,
    error: null,
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function createQueuedConversionJob(jobId: string): SourceConversionJobStatus {
  return {
    jobId,
    state: "queued",
    progress: {
      phase: "queued",
      percent: 0,
      message: "Conversion queued",
      processedFrames: 0,
      totalFrames: 0,
      extractedFrames: 0,
      upscaledFrames: 0,
      encodedFrames: 0,
      remuxedFrames: 0,
    },
    result: null,
    error: null,
  };
}

function ratioFromCounts(processed: number, total: number, completed: boolean): number {
  if (completed) {
    return 1;
  }
  if (total <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, processed / total));
}

function formatDurationProgress(units: number): string {
  return `${(units / 1000).toFixed(1)}s`;
}

function formatElapsedSeconds(seconds: number | null | undefined): string {
  if (!Number.isFinite(seconds ?? NaN) || (seconds ?? 0) < 0) {
    return "calculating";
  }

  const rounded = Math.round(seconds ?? 0);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const remainingSeconds = rounded % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m ${remainingSeconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${remainingSeconds}s`;
}

function formatClockTime(seconds: number | null | undefined): string {
  if (!Number.isFinite(seconds ?? NaN) || (seconds ?? 0) < 0) {
    return "0:00";
  }

  const rounded = Math.floor(seconds ?? 0);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const remainingSeconds = rounded % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function formatFramesPerSecond(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || (value ?? 0) <= 0) {
    return "calculating";
  }
  const resolved = value ?? 0;
  return `${resolved.toFixed(resolved >= 10 ? 1 : 2)} fps`;
}

function formatStageTimings(progress: PipelineProgress): string {
  const parts = [
    `extract ${formatElapsedSeconds(progress.extractStageSeconds)}`,
    `upscale ${formatElapsedSeconds(progress.upscaleStageSeconds)}`,
    `encode ${formatElapsedSeconds(progress.encodeStageSeconds)}`,
    `remux ${formatElapsedSeconds(progress.remuxStageSeconds)}`,
  ];
  return parts.join(", ");
}

function formatGpuMemory(usedBytes: number | null | undefined, totalBytes: number | null | undefined): string {
  if (!Number.isFinite(usedBytes ?? NaN) || (usedBytes ?? 0) <= 0) {
    return "unavailable";
  }
  if (!Number.isFinite(totalBytes ?? NaN) || (totalBytes ?? 0) <= 0) {
    return formatBytes(usedBytes ?? 0);
  }
  return `${formatBytes(usedBytes ?? 0)} / ${formatBytes(totalBytes ?? 0)}`;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }

  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatRelativeTime(timestamp: number): string {
  if (!Number.isFinite(timestamp) || timestamp <= 0) {
    return "unknown";
  }

  const deltaSeconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (deltaSeconds < 10) {
    return "just now";
  }
  if (deltaSeconds < 60) {
    return `${deltaSeconds}s ago`;
  }

  const deltaMinutes = Math.floor(deltaSeconds / 60);
  if (deltaMinutes < 60) {
    return `${deltaMinutes}m ago`;
  }

  const deltaHours = Math.floor(deltaMinutes / 60);
  if (deltaHours < 24) {
    return `${deltaHours}h ago`;
  }

  const deltaDays = Math.floor(deltaHours / 24);
  if (deltaDays < 30) {
    return `${deltaDays}d ago`;
  }

  const deltaMonths = Math.floor(deltaDays / 30);
  if (deltaMonths < 12) {
    return `${deltaMonths}mo ago`;
  }

  return `${Math.floor(deltaDays / 365)}y ago`;
}

function buildPipelineActivityTitle(progress: PipelineProgress): string {
  if (progress.phase === "extracting") {
    return "Extracting source frames";
  }
  if (progress.phase === "upscaling") {
    return progress.batchCount ? "Upscaling the current batch" : "Upscaling extracted frames";
  }
  if (progress.phase === "encoding") {
    return "Encoding upscaled frames";
  }
  if (progress.phase === "remuxing") {
    return "Remuxing audio and finishing output";
  }
  if (progress.phase === "completed") {
    return "Pipeline completed";
  }
  if (progress.phase === "failed") {
    return "Pipeline failed";
  }
  return "Preparing pipeline";
}

function buildPipelineActivityDetail(progress: PipelineProgress): string {
  const parts: string[] = [];
  const totalFrames = progress.totalFrames || 0;
  if (totalFrames > 0) {
    parts.push(`frames ${progress.processedFrames}/${totalFrames}`);
  }
  if (progress.segmentIndex && progress.segmentCount) {
    parts.push(`segment ${progress.segmentIndex}/${progress.segmentCount}`);
  }
  if (progress.segmentProcessedFrames !== null && progress.segmentProcessedFrames !== undefined && progress.segmentTotalFrames) {
    parts.push(`segment frames ${progress.segmentProcessedFrames}/${progress.segmentTotalFrames}`);
  }
  if (progress.batchIndex && progress.batchCount) {
    parts.push(`batch ${progress.batchIndex}/${progress.batchCount}`);
  }
  if ((progress.averageFramesPerSecond ?? 0) > 0) {
    parts.push(`avg ${formatFramesPerSecond(progress.averageFramesPerSecond)}`);
  }
  if ((progress.rollingFramesPerSecond ?? 0) > 0) {
    parts.push(`live ${formatFramesPerSecond(progress.rollingFramesPerSecond)}`);
  }
  if ((progress.estimatedRemainingSeconds ?? 0) > 0) {
    parts.push(`eta ${formatElapsedSeconds(progress.estimatedRemainingSeconds)}`);
  }
  return parts.length > 0 ? parts.join(" | ") : progress.message;
}

function buildProgressEventKey(progress: PipelineProgress): string {
  return [
    progress.phase,
    progress.percent,
    progress.message,
    progress.processedFrames,
    progress.extractedFrames,
    progress.upscaledFrames,
    progress.encodedFrames,
    progress.remuxedFrames,
    progress.segmentIndex ?? "",
    progress.segmentProcessedFrames ?? "",
    progress.batchIndex ?? "",
    progress.averageFramesPerSecond?.toFixed(3) ?? "",
    progress.rollingFramesPerSecond?.toFixed(3) ?? "",
  ].join("|");
}

function buildProgressEvent(progress: PipelineProgress, timestamp: number): ProgressEventEntry {
  return {
    key: buildProgressEventKey(progress),
    title: buildPipelineActivityTitle(progress),
    detail: buildPipelineActivityDetail(progress),
    percent: progress.percent,
    timestamp,
  };
}

function cleanupKindLabel(jobKind: string): string {
  return jobKind === "sourceConversion" ? "Conversion" : "Upscale";
}

function formatExactTimestamp(timestamp: number): string {
  if (!Number.isFinite(timestamp) || timestamp <= 0) {
    return "Unknown";
  }

  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function safeLocalStorageGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeLocalStorageSet(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore local storage failures.
  }
}

function parseCleanupFilter(value: string | null): CleanupJobFilter {
  return value === "running" || value === "succeeded" || value === "cancelled" || value === "failed" ? value : "all";
}

function parseCleanupSort(value: string | null): CleanupJobSort {
  return value === "newest" || value === "oldest" ? value : "largest";
}

function cleanupJobTotalBytes(job: TrackedJobEntry): number {
  return job.scratchSizeBytes + job.outputSizeBytes;
}

function matchesCleanupSearch(job: TrackedJobEntry, query: string): boolean {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return true;
  }

  return [
    job.label,
    job.state,
    job.phase,
    job.message,
    job.modelId ?? "",
    job.sourcePath ?? "",
    job.scratchPath ?? "",
    job.outputPath ?? "",
  ].some((value) => value.toLowerCase().includes(normalizedQuery));
}

function sortCleanupJobs(left: TrackedJobEntry, right: TrackedJobEntry, sortMode: CleanupJobSort): number {
  if (sortMode === "newest") {
    return right.updatedAt - left.updatedAt;
  }
  if (sortMode === "oldest") {
    return left.updatedAt - right.updatedAt;
  }

  const byteDelta = cleanupJobTotalBytes(right) - cleanupJobTotalBytes(left);
  return byteDelta !== 0 ? byteDelta : right.updatedAt - left.updatedAt;
}

function isManagedArtifactPath(path: string | null | undefined): boolean {
  if (!path) {
    return false;
  }

  return path.replace(/\\/g, "/").toLowerCase().includes("/artifacts/");
}

function shuffleModels(modelIds: ModelId[]): ModelId[] {
  const next = [...modelIds];
  for (let index = next.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    const current = next[index];
    next[index] = next[swapIndex] ?? current;
    next[swapIndex] = current;
  }
  return next;
}

export default function App() {
  const [modelId, setModelId] = useState<ModelId>("realesrgan-x4plus");
  const [outputMode, setOutputMode] = useState<OutputMode>("preserveAspect4k");
  const [qualityPreset, setQualityPreset] = useState<QualityPreset>("qualityBalanced");
  const [selectedGpuId, setSelectedGpuId] = useState<number | null>(null);
  const [aspectRatioPreset, setAspectRatioPreset] = useState<AspectRatioPreset>("16:9");
  const [customAspectWidthInput, setCustomAspectWidthInput] = useState<string>("16");
  const [customAspectHeightInput, setCustomAspectHeightInput] = useState<string>("9");
  const [resolutionBasis, setResolutionBasis] = useState<ResolutionBasis>("exact");
  const [targetWidthInput, setTargetWidthInput] = useState<string>("3840");
  const [targetHeightInput, setTargetHeightInput] = useState<string>("2160");
  const [cropRect, setCropRect] = useState<NormalizedCropRect | null>(null);
  const [codec, setCodec] = useState<VideoCodec>("h264");
  const [container, setContainer] = useState<OutputContainer>("mp4");
  const [tileSize, setTileSize] = useState<number>(0);
  const [crf, setCrf] = useState<number>(18);
  const [pytorchRunner, setPytorchRunner] = useState<PytorchRunner>(() => recommendedPytorchRunner(modelId));
  const [previewMode, setPreviewMode] = useState<boolean>(true);
  const [previewDurationInput, setPreviewDurationInput] = useState<string>("8");
  const [segmentDurationInput, setSegmentDurationInput] = useState<string>("10");
  const [source, setSource] = useState<SourceVideoSummary | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [appConfig, setAppConfig] = useState<AppConfig | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [pipelineJob, setPipelineJob] = useState<PipelineJobStatus | null>(null);
  const [pipelineProgressEvents, setPipelineProgressEvents] = useState<ProgressEventEntry[]>([]);
  const [lastPipelineProgressAt, setLastPipelineProgressAt] = useState<number | null>(null);
  const [sourceConversionJobId, setSourceConversionJobId] = useState<string | null>(null);
  const [sourceConversionJob, setSourceConversionJob] = useState<SourceConversionJobStatus | null>(null);
  const [sourceConversionMode, setSourceConversionMode] = useState<"preview" | "replace" | null>(null);
  const [sourceConversionSourcePath, setSourceConversionSourcePath] = useState<string | null>(null);
  const [previewPlaybackPath, setPreviewPlaybackPath] = useState<string | null>(null);
  const [uiNow, setUiNow] = useState<number>(() => Date.now());
  const [sourcePathStats, setSourcePathStats] = useState<PathStats | null>(null);
  const [outputPathStats, setOutputPathStats] = useState<PathStats | null>(null);
  const [workDirStats, setWorkDirStats] = useState<PathStats | null>(null);
  const [scratchSummary, setScratchSummary] = useState<ScratchStorageSummary | null>(null);
  const [managedJobs, setManagedJobs] = useState<ManagedJobSummary[]>([]);
  const [cleanupFilter, setCleanupFilter] = useState<CleanupJobFilter>(() => parseCleanupFilter(safeLocalStorageGet(CLEANUP_FILTER_STORAGE_KEY)));
  const [cleanupSearch, setCleanupSearch] = useState<string>(() => safeLocalStorageGet(CLEANUP_SEARCH_STORAGE_KEY) ?? "");
  const [cleanupSort, setCleanupSort] = useState<CleanupJobSort>(() => parseCleanupSort(safeLocalStorageGet(CLEANUP_SORT_STORAGE_KEY)));
  const [expandedCleanupJobIds, setExpandedCleanupJobIds] = useState<string[]>([]);
  const [isInputPanelOpen, setIsInputPanelOpen] = useState(true);
  const [isOutputPanelOpen, setIsOutputPanelOpen] = useState(true);
  const [isBlindPanelOpen, setIsBlindPanelOpen] = useState(false);
  const [isCleanupPanelOpen, setIsCleanupPanelOpen] = useState(false);
  const [blindComparison, setBlindComparison] = useState<BlindComparisonState | null>(null);
  const [status, setStatus] = useState<string>("Idle");
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [isSavingRating, setIsSavingRating] = useState(false);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [isCropEditing, setIsCropEditing] = useState(false);
  const [comparisonSampleId, setComparisonSampleId] = useState<string | null>(null);
  const [comparisonZoom, setComparisonZoom] = useState<number>(3);
  const [comparisonFocusX, setComparisonFocusX] = useState<number>(50);
  const [comparisonFocusY, setComparisonFocusY] = useState<number>(50);
  const [comparisonFocusPresetId, setComparisonFocusPresetId] = useState<string>(comparisonFocusPresets[0]?.id ?? "dithering");
  const [comparisonCurrentTime, setComparisonCurrentTime] = useState<number>(0);
  const [comparisonDuration, setComparisonDuration] = useState<number>(0);
  const [comparisonPlaying, setComparisonPlaying] = useState<boolean>(false);
  const [sourcePreviewPlaying, setSourcePreviewPlaying] = useState<boolean>(false);
  const [sourcePreviewCurrentTime, setSourcePreviewCurrentTime] = useState<number>(0);
  const [sourcePreviewDuration, setSourcePreviewDuration] = useState<number>(0);
  const [resolvedSourcePreviewUrl, setResolvedSourcePreviewUrl] = useState<string | null>(null);
  const previewRef = useRef<HTMLDivElement | null>(null);
  const sourcePreviewVideoRef = useRef<HTMLVideoElement | null>(null);
  const comparisonSourceVideoRef = useRef<HTMLVideoElement | null>(null);
  const comparisonOutputVideoRef = useRef<HTMLVideoElement | null>(null);
  const pipelineProgressSignatureRef = useRef<string | null>(null);
  const sourcePreviewAutoResumeRef = useRef(false);
  const resolvedSourcePreviewUrlRef = useRef<string | null>(null);

  const sizingOptions: OutputSizingOptions = {
    aspectRatioPreset,
    customAspectWidth: parsePositiveIntegerInput(customAspectWidthInput),
    customAspectHeight: parsePositiveIntegerInput(customAspectHeightInput),
    resolutionBasis,
    targetWidth: parsePositiveIntegerInput(targetWidthInput),
    targetHeight: parsePositiveIntegerInput(targetHeightInput),
    cropLeft: cropRect?.left ?? null,
    cropTop: cropRect?.top ?? null,
    cropWidth: cropRect?.width ?? null,
    cropHeight: cropRect?.height ?? null
  };

  const framing = source
    ? planOutputFraming({ width: source.width, height: source.height }, outputMode, sizingOptions)
    : null;
  const previewSourcePath = source ? ((previewPlaybackPath ?? source.previewPath) || source.path) : null;
  const previewSrc = resolvedSourcePreviewUrl;
  const sourcePreviewMimeType = previewMimeType(previewSourcePath);
  const resultPreviewSrc = result ? desktopApi.toPreviewSrc(result.outputPath) : null;
  const previewDurationSeconds = parsePositiveIntegerInput(previewDurationInput);
  const segmentDurationSeconds = parsePositiveIntegerInput(segmentDurationInput);
  const displayedWidth = resolutionBasis === "height" && framing ? String(framing.canvas.width) : targetWidthInput;
  const displayedHeight = resolutionBasis === "width" && framing ? String(framing.canvas.height) : targetHeightInput;
  const sourcePreviewSeekMax = Math.max(sourcePreviewDuration, 0.01);
  const previewCropRect = source && outputMode === "cropTo4k"
    ? resolveCropRect({ width: source.width, height: source.height }, sizingOptions)
    : null;
  const cropOverlayStyle = source && previewCropRect && outputMode === "cropTo4k"
    ? {
        width: `${previewCropRect.width * 100}%`,
        height: `${previewCropRect.height * 100}%`,
        left: `${previewCropRect.left * 100}%`,
        top: `${previewCropRect.top * 100}%`
      }
    : undefined;
  const aspectRatioValue = source ? resolveAspectRatio({ width: source.width, height: source.height }, sizingOptions) : 16 / 9;
  const isBlockingSourceConversionRunning = Boolean(sourceConversionJob && (sourceConversionJob.state === "queued" || sourceConversionJob.state === "running") && sourceConversionMode !== "preview");
  const activePrimaryJob = sourceConversionJob && isBlockingSourceConversionRunning
    ? sourceConversionJob
    : pipelineJob;
  const progressPercent = activePrimaryJob?.progress.percent ?? (result ? 100 : 0);
  const progressMessage = activePrimaryJob?.progress.message ?? status;
  const isPipelineRunning = pipelineJob?.state === "queued" || pipelineJob?.state === "running";
  const isSourceConversionRunning = sourceConversionJob?.state === "queued" || sourceConversionJob?.state === "running";
  const isBlindComparisonRunning = blindComparison?.state === "running";
  const selectedGpu = runtime?.availableGpus.find((gpu) => gpu.id === selectedGpuId) ?? null;
  const selectedModel = getModelDefinition(modelId);
  const selectedBackend = getBackendDefinition(selectedModel.backendId);
  const isSelectedModelImplemented = selectedModel.executionStatus === "runnable";
  const supportsPytorchRunner = selectedBackend.id === "pytorch-image-sr";
  const selectedModelRating = appConfig?.modelRatings[selectedModel.value]?.rating ?? null;
  const comparisonEntries = blindComparison?.entries.filter((entry) => Boolean(entry.status.result?.outputPath)) ?? [];
  const selectedComparisonEntry = comparisonEntries.find((entry) => entry.sampleId === comparisonSampleId) ?? comparisonEntries[0] ?? null;
  const selectedComparisonPreset = comparisonFocusPresets.find((preset) => preset.id === comparisonFocusPresetId) ?? comparisonFocusPresets[0];
  const usingFallbackPreviewClip = Boolean(source && source.previewPath !== source.path && !previewPlaybackPath);
  const canAutoUpgradePreview = Boolean(
    source
    && !supportsEmbeddedFullLengthPreview(source.container)
    && source.durationSeconds <= AUTO_PREVIEW_UPGRADE_MAX_DURATION_SECONDS
  );
  const previewUpgradeAvailable = Boolean(source && previewPlaybackPath && previewPlaybackPath !== source.previewPath);
  const previewUpgradePending = Boolean(
    source
    && !supportsEmbeddedFullLengthPreview(source.container)
    && canAutoUpgradePreview
    && !previewUpgradeAvailable
    && sourceConversionMode === "preview"
    && isSourceConversionRunning
  );
  const isRunDisabled = isBusy || !source || isBlindComparisonRunning || isPipelineRunning || isBlockingSourceConversionRunning || !isSelectedModelImplemented;
  const isBlindComparisonDisabled = isBusy || !source || isBlindComparisonRunning || isPipelineRunning || isBlockingSourceConversionRunning || blindComparisonCandidates.length < 2;
  const nowTimestamp = uiNow;
  const trackedJobCandidates: Array<TrackedJobEntry | null> = [
    sourceConversionJob ? {
      id: sourceConversionJob.jobId,
      jobKind: "sourceConversion",
      label: "Source Conversion",
      state: sourceConversionJob.state,
      phase: sourceConversionJob.progress.phase,
      progress: sourceConversionJob.progress,
      modelId: null,
      codec: null,
      container: sourceConversionJob.result?.container ?? "mp4",
      recordedCount: sourceConversionJob.progress.totalFrames,
      message: sourceConversionJob.progress.message,
      updatedAt: nowTimestamp,
      sourcePath: source?.path ?? null,
      scratchPath: null,
      scratchSizeBytes: 0,
      outputPath: sourceConversionJob.result?.path ?? null,
      outputSizeBytes: sourceConversionJob.progress.outputSizeBytes ?? (sourceConversionJob.result ? sourcePathStats?.sizeBytes ?? 0 : 0),
      onStop: isSourceConversionRunning ? () => {
        void cancelSourceConversion();
      } : null,
      onClearScratch: null as null | (() => void),
      onDeleteOutput: sourceConversionJob.result?.path && isManagedArtifactPath(sourceConversionJob.result.path)
        ? () => void deleteManagedArtifact(sourceConversionJob.result!.path, "Converted input", clearLoadedInput)
        : null,
    } : null,
    pipelineJob ? {
      id: pipelineJob.jobId,
      jobKind: "pipeline",
      label: result ? "Upscale Export" : "Upscale Job",
      state: pipelineJob.state,
      phase: pipelineJob.progress.phase,
      progress: pipelineJob.progress,
      modelId: modelId,
      codec: result?.codec ?? codec,
      container: result?.container ?? container,
      recordedCount: pipelineJob.progress.totalFrames,
      message: pipelineJob.progress.message,
      updatedAt: nowTimestamp,
      sourcePath: source?.path ?? null,
      scratchPath: result?.workDir ?? null,
      scratchSizeBytes: pipelineJob.progress.scratchSizeBytes ?? workDirStats?.sizeBytes ?? 0,
      outputPath: result?.outputPath ?? outputPath ?? null,
      outputSizeBytes: pipelineJob.progress.outputSizeBytes ?? outputPathStats?.sizeBytes ?? 0,
      onStop: isPipelineRunning ? () => {
        void cancelPipeline();
      } : null,
      onClearScratch: result?.workDir && isManagedArtifactPath(result.workDir)
        ? () => void deleteManagedArtifact(result.workDir, "Job scratch", () => setWorkDirStats(null))
        : null,
      onDeleteOutput: result?.outputPath && isManagedArtifactPath(result.outputPath)
        ? () => void deleteManagedArtifact(result.outputPath, "Job output", clearCurrentOutputSelection)
        : null,
    } : null,
  ];
  const trackedJobs = trackedJobCandidates.filter((entry): entry is TrackedJobEntry => entry !== null);
  const historicalJobs = managedJobs.map<TrackedJobEntry>((job) => ({
    id: job.jobId,
    jobKind: job.jobKind,
    label: job.label,
    state: job.state as TrackedJobEntry["state"],
    phase: job.progress.phase,
    progress: job.progress,
    modelId: job.modelId,
    codec: job.codec,
    container: job.container,
    recordedCount: job.recordedCount,
    message: job.progress.message,
    updatedAt: Number(job.updatedAt) || 0,
    sourcePath: job.sourcePath,
    scratchPath: job.scratchPath,
    scratchSizeBytes: job.progress.scratchSizeBytes ?? job.scratchStats?.sizeBytes ?? 0,
    outputPath: job.outputPath,
    outputSizeBytes: job.progress.outputSizeBytes ?? job.outputStats?.sizeBytes ?? 0,
    onStop: null,
    onClearScratch: job.scratchPath && isManagedArtifactPath(job.scratchPath)
      ? () => void deleteManagedArtifact(job.scratchPath!, `${job.label} scratch`, () => {})
      : null,
    onDeleteOutput: job.outputPath && isManagedArtifactPath(job.outputPath)
      ? () => void deleteManagedArtifact(job.outputPath!, `${job.label} output`, () => {})
      : null,
  }));
  const cleanupJobsById = new Map<string, TrackedJobEntry>();
  for (const job of historicalJobs) {
    cleanupJobsById.set(job.id, job);
  }
  for (const job of trackedJobs) {
    const existing = cleanupJobsById.get(job.id);
    cleanupJobsById.set(job.id, {
      ...existing,
      ...job,
      message: job.message || existing?.message || "",
      progress: job.progress,
      updatedAt: Math.max(job.updatedAt, existing?.updatedAt ?? 0),
      sourcePath: job.sourcePath ?? existing?.sourcePath ?? null,
      scratchPath: job.scratchPath ?? existing?.scratchPath ?? null,
      scratchSizeBytes: job.scratchSizeBytes || existing?.scratchSizeBytes || job.progress.scratchSizeBytes || 0,
      outputPath: job.outputPath ?? existing?.outputPath ?? null,
      outputSizeBytes: job.outputSizeBytes || existing?.outputSizeBytes || job.progress.outputSizeBytes || 0,
      onStop: job.onStop ?? existing?.onStop ?? null,
      onClearScratch: job.onClearScratch ?? existing?.onClearScratch ?? null,
      onDeleteOutput: job.onDeleteOutput ?? existing?.onDeleteOutput ?? null,
    });
  }
  const cleanupJobs = Array.from(cleanupJobsById.values()).sort((left, right) => sortCleanupJobs(left, right, cleanupSort));
  const cleanupStateCounts = {
    all: cleanupJobs.length,
    running: cleanupJobs.filter((job) => job.state === "queued" || job.state === "running").length,
    succeeded: cleanupJobs.filter((job) => job.state === "succeeded").length,
    cancelled: cleanupJobs.filter((job) => job.state === "cancelled").length,
    failed: cleanupJobs.filter((job) => job.state === "failed").length,
  };
  const filteredCleanupJobs = cleanupJobs.filter((job) => {
    if (cleanupFilter === "all") {
      return matchesCleanupSearch(job, cleanupSearch);
    }
    if (cleanupFilter === "running") {
      return (job.state === "queued" || job.state === "running") && matchesCleanupSearch(job, cleanupSearch);
    }
    return job.state === cleanupFilter && matchesCleanupSearch(job, cleanupSearch);
  });
  const hasActiveCleanupJobs = cleanupJobs.some((job) => job.state === "queued" || job.state === "running");
  const pipelinePhaseBars = pipelineJob ? [
    {
      id: "extract",
      label: "Extract",
      value: ratioFromCounts(pipelineJob.progress.extractedFrames, pipelineJob.progress.totalFrames, pipelineJob.state === "succeeded"),
      summary: `${pipelineJob.progress.extractedFrames}/${pipelineJob.progress.totalFrames || "?"}`,
    },
    {
      id: "upscale",
      label: "Upscale",
      value: ratioFromCounts(pipelineJob.progress.upscaledFrames, pipelineJob.progress.totalFrames, pipelineJob.state === "succeeded"),
      summary: `${pipelineJob.progress.upscaledFrames}/${pipelineJob.progress.totalFrames || "?"}`,
    },
    {
      id: "encode",
      label: "Encode",
      value: ratioFromCounts(pipelineJob.progress.encodedFrames, pipelineJob.progress.totalFrames, pipelineJob.state === "succeeded"),
      summary: `${pipelineJob.progress.encodedFrames}/${pipelineJob.progress.totalFrames || "?"}`,
    },
    {
      id: "remux",
      label: "Remux",
      value: ratioFromCounts(pipelineJob.progress.remuxedFrames, pipelineJob.progress.totalFrames, pipelineJob.state === "succeeded"),
      summary: `${pipelineJob.progress.remuxedFrames}/${pipelineJob.progress.totalFrames || "?"}`,
    },
  ] : [];
  const progressScratchSizeBytes = pipelineJob?.progress.scratchSizeBytes ?? workDirStats?.sizeBytes ?? 0;
  const progressOutputSizeBytes = pipelineJob?.progress.outputSizeBytes ?? outputPathStats?.sizeBytes ?? 0;
  const pipelineActivityTitle = pipelineJob ? buildPipelineActivityTitle(pipelineJob.progress) : null;
  const pipelineActivityDetail = pipelineJob ? buildPipelineActivityDetail(pipelineJob.progress) : null;
  const pipelineLastUpdateLabel = lastPipelineProgressAt ? formatRelativeTime(lastPipelineProgressAt) : "waiting for first update";

  useEffect(() => {
    if (!isPipelineRunning && !isSourceConversionRunning) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      setUiNow(Date.now());
    }, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isPipelineRunning, isSourceConversionRunning]);

  useEffect(() => {
    if (!pipelineJob) {
      pipelineProgressSignatureRef.current = null;
      return;
    }

    const timestamp = Date.now();
    setLastPipelineProgressAt(timestamp);

    const nextEntry = buildProgressEvent(pipelineJob.progress, timestamp);
    if (pipelineProgressSignatureRef.current === nextEntry.key) {
      return;
    }

    pipelineProgressSignatureRef.current = nextEntry.key;
    setPipelineProgressEvents((current) => {
      const next = [...current, nextEntry];
      return next.slice(-8);
    });
  }, [pipelineJob]);

  useEffect(() => {
    if (comparisonEntries.length === 0) {
      if (comparisonSampleId !== null) {
        setComparisonSampleId(null);
      }
      return;
    }

    if (!comparisonSampleId || !comparisonEntries.some((entry) => entry.sampleId === comparisonSampleId)) {
      setComparisonSampleId(comparisonEntries[0]?.sampleId ?? null);
    }
  }, [comparisonEntries, comparisonSampleId]);

  useEffect(() => {
    const preset = comparisonFocusPresets.find((entry) => entry.id === comparisonFocusPresetId);
    if (!preset) {
      return;
    }

    setComparisonFocusX(preset.focusX);
    setComparisonFocusY(preset.focusY);
  }, [comparisonFocusPresetId]);

  useEffect(() => {
    let cancelled = false;

    async function loadAppConfig(): Promise<void> {
      try {
        const config = await desktopApi.getAppConfig();
        if (!cancelled) {
          setAppConfig(config);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      }
    }

    void loadAppConfig();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!runtime) {
      return;
    }

    if (selectedGpuId !== null && runtime.availableGpus.some((gpu) => gpu.id === selectedGpuId)) {
      return;
    }

    if (runtime.defaultGpuId !== null) {
      setSelectedGpuId(runtime.defaultGpuId);
      return;
    }

    setSelectedGpuId(runtime.availableGpus[0]?.id ?? null);
  }, [runtime, selectedGpuId]);

  useEffect(() => {
    if (!source || outputMode !== "cropTo4k") {
      setCropRect(null);
      setIsCropEditing(false);
      return;
    }

    setCropRect(defaultCropRect(
      { width: source.width, height: source.height },
      {
        ...sizingOptions,
        cropLeft: null,
        cropTop: null,
        cropWidth: null,
        cropHeight: null
      }
    ));
  }, [source?.path, source?.width, source?.height, outputMode, aspectRatioPreset, customAspectWidthInput, customAspectHeightInput]);

  useEffect(() => {
    if (!dragState) {
      return undefined;
    }

    const activeDragState = dragState;

    function handleMouseMove(event: MouseEvent): void {
      const preview = previewRef.current;
      if (!preview || !source) {
        return;
      }

      const bounds = preview.getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) {
        return;
      }

      const deltaX = (event.clientX - activeDragState.startX) / bounds.width;
      const deltaY = (event.clientY - activeDragState.startY) / bounds.height;
      if (activeDragState.handle === "move") {
        setCropRect(clampCropRect({
          ...activeDragState.startRect,
          left: activeDragState.startRect.left + deltaX,
          top: activeDragState.startRect.top + deltaY
        }));
        return;
      }

      setCropRect(resizeCropRect(
        activeDragState.startRect,
        activeDragState.handle,
        deltaX,
        deltaY,
        aspectRatioValue,
        source.width / source.height,
      ));
    }

    function handleMouseUp(): void {
      setDragState(null);
    }

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [aspectRatioValue, dragState, source]);

  useEffect(() => {
    if (!activeJobId) {
      return undefined;
    }

    const currentJobId = activeJobId;
    let cancelled = false;

    async function pollJob(): Promise<void> {
      try {
        const nextJob = await desktopApi.getPipelineJob(currentJobId);
        if (cancelled) {
          return;
        }

        setPipelineJob(nextJob);
        setStatus(nextJob.progress.message);
        if (nextJob.state === "succeeded" && nextJob.result) {
          setResult(nextJob.result);
          setOutputPath(nextJob.result.outputPath);
          setActiveJobId(null);
          setStatus("Pipeline completed.");
        }

        if (nextJob.state === "failed") {
          setError(nextJob.error ?? nextJob.progress.message);
          setActiveJobId(null);
          setStatus("Pipeline failed.");
        }

        if (nextJob.state === "cancelled") {
          setResult(null);
          setActiveJobId(null);
          setStatus("Pipeline cancelled.");
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
          setActiveJobId(null);
          setStatus("Pipeline failed.");
        }
      }
    }

    void pollJob();
    const intervalId = window.setInterval(() => {
      void pollJob();
    }, 1000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [activeJobId]);

  useEffect(() => {
    setPytorchRunner(recommendedPytorchRunner(modelId));
  }, [modelId]);

  useEffect(() => {
    sourcePreviewAutoResumeRef.current = sourcePreviewAutoResumeRef.current || sourcePreviewPlaying;
    setSourcePreviewPlaying(false);
    setSourcePreviewCurrentTime(0);
    setSourcePreviewDuration(source?.durationSeconds ?? 0);
  }, [previewSourcePath]);

  useEffect(() => {
    let cancelled = false;

    async function resolvePreviewUrl(): Promise<void> {
      if (!previewSourcePath) {
        replaceResolvedSourcePreviewUrl(null);
        return;
      }

      try {
        const nextUrl = await desktopApi.loadPreviewUrl(previewSourcePath);
        if (cancelled) {
          if (nextUrl.startsWith("blob:")) {
            URL.revokeObjectURL(nextUrl);
          }
          return;
        }

        replaceResolvedSourcePreviewUrl(nextUrl);
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
          setStatus("Preview load failed.");
          replaceResolvedSourcePreviewUrl(null);
        }
      }
    }

    void resolvePreviewUrl();

    return () => {
      cancelled = true;
    };
  }, [previewSourcePath]);

  useEffect(() => () => {
    const currentUrl = resolvedSourcePreviewUrlRef.current;
    if (currentUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(currentUrl);
    }
  }, []);

  function replaceResolvedSourcePreviewUrl(nextUrl: string | null): void {
    const currentUrl = resolvedSourcePreviewUrlRef.current;
    if (currentUrl?.startsWith("blob:") && currentUrl !== nextUrl) {
      URL.revokeObjectURL(currentUrl);
    }
    resolvedSourcePreviewUrlRef.current = nextUrl;
    setResolvedSourcePreviewUrl(nextUrl);
  }

  async function resolveSourcePreviewUrl(path: string | null): Promise<void> {
    if (!path) {
      replaceResolvedSourcePreviewUrl(null);
      return;
    }

    try {
      const nextUrl = await desktopApi.loadPreviewUrl(path);
      replaceResolvedSourcePreviewUrl(nextUrl);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Preview load failed.");
      replaceResolvedSourcePreviewUrl(null);
    }
  }

  useEffect(() => {
    if (!sourceConversionJobId) {
      return undefined;
    }

    const currentJobId = sourceConversionJobId;
    let cancelled = false;

    async function pollJob(): Promise<void> {
      try {
        const nextJob = await desktopApi.getSourceConversionJob(currentJobId);
        if (cancelled) {
          return;
        }

        setSourceConversionJob(nextJob);
        setStatus(nextJob.progress.message);
        if (nextJob.state === "succeeded" && nextJob.result) {
          const conversionMode = sourceConversionMode;
          const conversionSourcePath = sourceConversionSourcePath;
          const isCurrentSource = source?.path === conversionSourcePath;

          if (conversionMode === "preview") {
            if (isCurrentSource) {
              setPreviewPlaybackPath(nextJob.result.path);
              void resolveSourcePreviewUrl(nextJob.result.path);
              setStatus("Full-length preview ready.");
            }
            setSourceConversionJobId(null);
          } else {
            setSource(nextJob.result);
            setPreviewPlaybackPath(null);
            void resolveSourcePreviewUrl(nextJob.result.previewPath || nextJob.result.path);
            setOutputPath(defaultOutputPath(nextJob.result, container, modelId));
            setResult(null);
            setPipelineJob(null);
            setActiveJobId(null);
            setBlindComparison(null);
            setIsCropEditing(false);
            setComparisonSampleId(null);
            setComparisonCurrentTime(0);
            setComparisonDuration(0);
            setComparisonPlaying(false);
            setSourceConversionJobId(null);
            setStatus("Source converted to MP4.");
          }
        }

        if (nextJob.state === "failed") {
          setError(nextJob.error ?? nextJob.progress.message);
          setSourceConversionJobId(null);
          setStatus("Source conversion failed.");
        }

        if (nextJob.state === "cancelled") {
          setSourceConversionJobId(null);
          setStatus("Source conversion cancelled.");
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
          setSourceConversionJobId(null);
          setStatus("Source conversion failed.");
        }
      }
    }

    void pollJob();
    const intervalId = window.setInterval(() => {
      void pollJob();
    }, 500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [container, modelId, source, sourceConversionJobId, sourceConversionMode, sourceConversionSourcePath]);

  async function loadManagedArtifacts(): Promise<{
    scratchSummary: ScratchStorageSummary;
    managedJobs: ManagedJobSummary[];
    sourcePathStats: PathStats | null;
    outputPathStats: PathStats | null;
    workDirStats: PathStats | null;
  }> {
    const [nextScratch, nextManagedJobs, nextSource, nextOutput, nextWorkDir] = await Promise.all([
      desktopApi.getScratchStorageSummary(),
      desktopApi.listManagedJobs(),
      source ? desktopApi.getPathStats(source.path) : Promise.resolve(null),
      result ? desktopApi.getPathStats(result.outputPath) : Promise.resolve(outputPath ? desktopApi.getPathStats(outputPath) : null),
      result ? desktopApi.getPathStats(result.workDir) : Promise.resolve(null),
    ]);

    return {
      scratchSummary: nextScratch,
      managedJobs: nextManagedJobs,
      sourcePathStats: nextSource,
      outputPathStats: nextOutput,
      workDirStats: nextWorkDir,
    };
  }

  function applyManagedArtifacts(snapshot: {
    scratchSummary: ScratchStorageSummary;
    managedJobs: ManagedJobSummary[];
    sourcePathStats: PathStats | null;
    outputPathStats: PathStats | null;
    workDirStats: PathStats | null;
  }): void {
    setScratchSummary(snapshot.scratchSummary);
    setManagedJobs(snapshot.managedJobs);
    setSourcePathStats(snapshot.sourcePathStats);
    setOutputPathStats(snapshot.outputPathStats);
    setWorkDirStats(snapshot.workDirStats);
  }

  useEffect(() => {
    let disposed = false;

    async function refreshStorage(): Promise<void> {
      try {
        const snapshot = await loadManagedArtifacts();

        if (disposed) {
          return;
        }

        applyManagedArtifacts(snapshot);
      } catch {
        if (!disposed) {
          setScratchSummary(null);
          setManagedJobs([]);
        }
      }
    }

    void refreshStorage();
    return () => {
      disposed = true;
    };
  }, [outputPath, result, source, sourceConversionJob]);

  useEffect(() => {
    safeLocalStorageSet(CLEANUP_FILTER_STORAGE_KEY, cleanupFilter);
  }, [cleanupFilter]);

  useEffect(() => {
    safeLocalStorageSet(CLEANUP_SEARCH_STORAGE_KEY, cleanupSearch);
  }, [cleanupSearch]);

  useEffect(() => {
    safeLocalStorageSet(CLEANUP_SORT_STORAGE_KEY, cleanupSort);
  }, [cleanupSort]);

  useEffect(() => {
    setExpandedCleanupJobIds((current) => current.filter((jobId) => cleanupJobs.some((job) => job.id === jobId)));
  }, [cleanupJobs]);

  async function ensureRuntime(): Promise<void> {
    setStatus("Preparing runtime assets...");
    const nextRuntime = await desktopApi.ensureRuntimeAssets();
    setRuntime(nextRuntime);
  }

  function refreshComparisonDuration(): void {
    const sourceDuration = comparisonSourceVideoRef.current?.duration;
    const outputDuration = comparisonOutputVideoRef.current?.duration;
    const durations = [sourceDuration, outputDuration].filter((value): value is number => Number.isFinite(value ?? NaN));
    setComparisonDuration(durations.length > 0 ? Math.max(...durations) : 0);
  }

  function syncComparisonTime(nextTime: number): void {
    const sourceVideo = comparisonSourceVideoRef.current;
    const outputVideo = comparisonOutputVideoRef.current;
    const syncTargets = [sourceVideo, outputVideo];
    for (const video of syncTargets) {
      if (!video || !Number.isFinite(video.duration)) {
        continue;
      }

      const clamped = Math.max(0, Math.min(nextTime, video.duration || nextTime));
      if (Math.abs(video.currentTime - clamped) > 0.05) {
        video.currentTime = clamped;
      }
    }
    setComparisonCurrentTime(nextTime);
  }

  function handleComparisonLoadedMetadata(): void {
    refreshComparisonDuration();
  }

  function handleComparisonSourceTimeUpdate(): void {
    const sourceVideo = comparisonSourceVideoRef.current;
    if (!sourceVideo) {
      return;
    }

    const nextTime = sourceVideo.currentTime;
    setComparisonCurrentTime(nextTime);
    const outputVideo = comparisonOutputVideoRef.current;
    if (outputVideo && Number.isFinite(outputVideo.duration) && Math.abs(outputVideo.currentTime - nextTime) > 0.08) {
      outputVideo.currentTime = Math.max(0, Math.min(nextTime, outputVideo.duration || nextTime));
    }
  }

  async function toggleComparisonPlayback(): Promise<void> {
    const sourceVideo = comparisonSourceVideoRef.current;
    const outputVideo = comparisonOutputVideoRef.current;
    if (!sourceVideo || !outputVideo) {
      return;
    }

    if (comparisonPlaying) {
      sourceVideo.pause();
      outputVideo.pause();
      setComparisonPlaying(false);
      return;
    }

    await Promise.all([sourceVideo.play(), outputVideo.play()]);
    setComparisonPlaying(true);
  }

  function restartComparisonPlayback(): void {
    syncComparisonTime(0);
  }

  async function openMediaInDefaultApp(path: string): Promise<void> {
    try {
      setError(null);
      await desktopApi.openPathInDefaultApp(path);
      setStatus("Opened media in the default video app.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to open media in the default app.");
    }
  }

  async function toggleSourcePreviewPlayback(): Promise<void> {
    const video = sourcePreviewVideoRef.current;
    if (!video) {
      return;
    }

    if (!video.paused && !video.ended) {
      sourcePreviewAutoResumeRef.current = false;
      video.pause();
      setSourcePreviewPlaying(false);
      return;
    }

    sourcePreviewAutoResumeRef.current = true;
    try {
      await video.play();
      setSourcePreviewPlaying(true);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : String(caught);
      if (message.toLowerCase().includes("media was removed from the document")) {
        setStatus("Preview upgraded. Resuming playback...");
        return;
      }

      sourcePreviewAutoResumeRef.current = false;
      setError(message);
      setStatus("Preview playback failed.");
    }
  }

  function restartSourcePreviewPlayback(): void {
    const video = sourcePreviewVideoRef.current;
    if (!video) {
      return;
    }

    sourcePreviewAutoResumeRef.current = true;
    video.currentTime = 0;
    setSourcePreviewCurrentTime(0);
  }

  function handleSourcePreviewLoadedMetadata(): void {
    const video = sourcePreviewVideoRef.current;
    if (!video) {
      return;
    }

    if (Number.isFinite(video.duration) && video.duration > 0) {
      setSourcePreviewDuration(video.duration);
    }

    if (sourcePreviewAutoResumeRef.current && (video.paused || video.ended)) {
      void video.play().then(() => {
        setSourcePreviewPlaying(true);
      }).catch((caught) => {
        sourcePreviewAutoResumeRef.current = false;
        setError(caught instanceof Error ? caught.message : String(caught));
        setStatus("Preview playback failed.");
      });
    }
  }

  function handleSourcePreviewTimeUpdate(): void {
    const video = sourcePreviewVideoRef.current;
    if (!video) {
      return;
    }

    setSourcePreviewCurrentTime(video.currentTime);
    if (Number.isFinite(video.duration) && video.duration > 0) {
      setSourcePreviewDuration(video.duration);
    }
  }

  function seekSourcePreview(nextTime: number): void {
    const video = sourcePreviewVideoRef.current;
    const clamped = Math.max(0, Math.min(nextTime, sourcePreviewSeekMax));
    setSourcePreviewCurrentTime(clamped);
    if (!video) {
      return;
    }

    video.currentTime = clamped;
  }

  async function chooseOutputFile(): Promise<string | null> {
    const selected = await desktopApi.selectOutputFile(defaultOutputPath(source, container, modelId), container);
    if (!selected) {
      return null;
    }

    const normalized = normalizeOutputPath(selected, container);
    setOutputPath(normalized);
    return normalized;
  }

  function buildPipelineRequest(targetModelId: ModelId, targetOutputPath: string, quickPreview: boolean, quickPreviewSeconds: number | null): RealesrganJobRequest {
    if (!source) {
      throw new Error("Select a source video before starting a pipeline.");
    }

    return {
      sourcePath: source.path,
      modelId: targetModelId,
      outputMode,
      qualityPreset,
      pytorchRunner: supportsPytorchRunner ? pytorchRunner : "torch",
      gpuId: selectedGpuId,
      aspectRatioPreset,
      customAspectWidth: sizingOptions.customAspectWidth,
      customAspectHeight: sizingOptions.customAspectHeight,
      resolutionBasis,
      targetWidth: sizingOptions.targetWidth,
      targetHeight: sizingOptions.targetHeight,
      cropLeft: sizingOptions.cropLeft,
      cropTop: sizingOptions.cropTop,
      cropWidth: sizingOptions.cropWidth,
      cropHeight: sizingOptions.cropHeight,
      previewMode: quickPreview,
      previewDurationSeconds: quickPreview ? quickPreviewSeconds : null,
      segmentDurationSeconds: quickPreview ? null : segmentDurationSeconds,
      outputPath: targetOutputPath,
      codec,
      container,
      tileSize,
      fp16: false,
      crf,
    };
  }

  async function selectVideo(): Promise<void> {
    try {
      setIsBusy(true);
      setError(null);
      await ensureRuntime();
      const selected = await desktopApi.selectVideoFile();
      if (!selected) {
        setStatus("Selection cancelled.");
        return;
      }

      setStatus("Probing source video...");
      const summary = await desktopApi.probeSourceVideo(selected);
      setSource(summary);
      await resolveSourcePreviewUrl(summary.previewPath || summary.path);
      setPreviewPlaybackPath(null);
      setOutputPath(defaultOutputPath(summary, container, modelId));
      setResult(null);
      setPipelineJob(null);
      setActiveJobId(null);
      setSourceConversionJob(null);
      setSourceConversionJobId(null);
      setSourceConversionMode(null);
      setSourceConversionSourcePath(null);
      setBlindComparison(null);
      setPipelineProgressEvents([]);
      setLastPipelineProgressAt(null);
      pipelineProgressSignatureRef.current = null;
      setIsCropEditing(false);
      setComparisonSampleId(null);
      setComparisonCurrentTime(0);
      setComparisonDuration(0);
      setComparisonPlaying(false);
      if (supportsEmbeddedFullLengthPreview(summary.container)) {
        setStatus("Source loaded.");
      } else {
        if (summary.durationSeconds <= AUTO_PREVIEW_UPGRADE_MAX_DURATION_SECONDS) {
          setStatus("Source loaded. Preparing full-length preview in the background...");
          void startSourceConversionJob(summary.path, "preview", { background: true });
        } else {
          setStatus("Source loaded with a lightweight fallback preview clip.");
        }
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Source load failed.");
    } finally {
      setIsBusy(false);
    }
  }

  async function startSourceConversionJob(
    sourcePath: string,
    mode: "preview" | "replace",
    options?: { background?: boolean },
  ): Promise<void> {
    const isBackground = options?.background ?? false;
    if (!isBackground) {
      setIsBusy(true);
    }

    try {
      setError(null);
      if (mode === "replace") {
        setStatus("Converting source to MP4...");
      } else {
        setStatus("Preparing full-length preview in the background...");
      }

      await ensureRuntime();
      const jobId = await desktopApi.startSourceConversionToMp4(sourcePath);
      setSourceConversionMode(mode);
      setSourceConversionSourcePath(sourcePath);
      setSourceConversionJob(createQueuedConversionJob(jobId));
      setSourceConversionJobId(jobId);
      if (mode === "replace") {
        setStatus("Source conversion queued.");
      } else {
        setStatus("Using fallback preview while the full-length preview prepares.");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      if (mode === "replace") {
        setStatus("Source conversion failed.");
      } else {
        setStatus("Playable preview preparation failed.");
      }
    } finally {
      if (!isBackground) {
        setIsBusy(false);
      }
    }
  }

  async function convertCurrentSourceToMp4(): Promise<void> {
    if (!source) {
      return;
    }

    await startSourceConversionJob(source.path, "replace");
  }

  async function runPipeline(): Promise<void> {
    if (!source) {
      setError("Select a source video first.");
      return;
    }

    if (!isSelectedModelImplemented) {
      setError(`${selectedModel.label} is cataloged but not implemented yet.`);
      setStatus("Selected model is not implemented yet.");
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      const selectedOutputPath = outputPath ?? await chooseOutputFile();
      if (!selectedOutputPath) {
        setStatus("Output selection cancelled.");
        return;
      }

      await ensureRuntime();
      setStatus("Starting Real-ESRGAN pipeline...");
      const jobId = await desktopApi.startPipeline(
        buildPipelineRequest(modelId, selectedOutputPath, previewMode, previewMode ? previewDurationSeconds : null)
      );
      setResult(null);
      setPipelineProgressEvents([]);
      setLastPipelineProgressAt(null);
      pipelineProgressSignatureRef.current = null;
      setPipelineJob(createQueuedJob(jobId));
      setActiveJobId(jobId);
      setStatus("Job queued.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Pipeline failed.");
    } finally {
      setIsBusy(false);
    }
  }

  async function cancelPipeline(): Promise<void> {
    if (!activeJobId) {
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await desktopApi.cancelPipelineJob(activeJobId);
      setStatus("Stopping pipeline...");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to stop pipeline.");
    } finally {
      setIsBusy(false);
    }
  }

  async function cancelSourceConversion(): Promise<void> {
    if (!sourceConversionJobId) {
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await desktopApi.cancelSourceConversionJob(sourceConversionJobId);
      setSourceConversionMode(null);
      setSourceConversionSourcePath(null);
      setStatus("Stopping source conversion...");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to stop source conversion.");
    } finally {
      setIsBusy(false);
    }
  }

  function clearLoadedInput(): void {
    setSource(null);
    setSourcePathStats(null);
    setSourceConversionJob(null);
    setSourceConversionJobId(null);
    setSourceConversionMode(null);
    setSourceConversionSourcePath(null);
    setPreviewPlaybackPath(null);
    setResult(null);
    setOutputPath(null);
    setOutputPathStats(null);
    setWorkDirStats(null);
    setPipelineJob(null);
    setPipelineProgressEvents([]);
    setLastPipelineProgressAt(null);
    pipelineProgressSignatureRef.current = null;
    setActiveJobId(null);
    setBlindComparison(null);
    setStatus("Input cleared from the workspace.");
  }

  function clearCurrentOutputSelection(): void {
    setResult(null);
    setPipelineJob(null);
    setPipelineProgressEvents([]);
    setLastPipelineProgressAt(null);
    pipelineProgressSignatureRef.current = null;
    setActiveJobId(null);
    setOutputPathStats(null);
    setWorkDirStats(null);
    setStatus("Output cleared from the workspace.");
  }

  async function deleteManagedArtifact(path: string, label: string, afterDelete: () => void): Promise<void> {
    try {
      setIsBusy(true);
      setError(null);
      await desktopApi.deleteManagedPath(path);
      afterDelete();
      applyManagedArtifacts(await loadManagedArtifacts());
      setStatus(`${label} deleted.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus(`Failed to delete ${label.toLowerCase()}.`);
    } finally {
      setIsBusy(false);
    }
  }

  function toggleCleanupJobExpanded(jobId: string): void {
    setExpandedCleanupJobIds((current) => current.includes(jobId) ? current.filter((value) => value !== jobId) : [...current, jobId]);
  }

  async function deleteManagedArtifacts(paths: string[], label: string): Promise<void> {
    if (paths.length === 0) {
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      for (const path of paths) {
        await desktopApi.deleteManagedPath(path);
      }
      applyManagedArtifacts(await loadManagedArtifacts());
      setStatus(`${label} deleted.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus(`Failed to delete ${label.toLowerCase()}.`);
    } finally {
      setIsBusy(false);
    }
  }

  async function runBulkCleanup(kind: "scratch" | "output" | "all"): Promise<void> {
    if (hasActiveCleanupJobs) {
      setStatus("Bulk cleanup is disabled while jobs are running.");
      return;
    }

    const scratchTargets = filteredCleanupJobs
      .map((job) => ({ path: job.scratchPath, bytes: job.scratchSizeBytes }))
      .filter((entry): entry is { path: string; bytes: number } => Boolean(entry.path));
    const outputTargets = filteredCleanupJobs
      .map((job) => ({ path: job.outputPath, bytes: job.outputSizeBytes }))
      .filter((entry): entry is { path: string; bytes: number } => Boolean(entry.path));

    const selectedTargets = kind === "scratch"
      ? scratchTargets
      : kind === "output"
        ? outputTargets
        : [...scratchTargets, ...outputTargets];

    if (selectedTargets.length === 0) {
      setStatus("No matching artifacts are available for bulk cleanup.");
      return;
    }

    const totalBytes = selectedTargets.reduce((sum, entry) => sum + entry.bytes, 0);
    const confirmed = window.confirm(
      `Delete ${selectedTargets.length} ${kind === "all" ? "artifacts" : kind === "scratch" ? "scratch paths" : "output files"} across ${filteredCleanupJobs.length} filtered jobs?\n\nImpacted size: ${formatBytes(totalBytes)}`
    );

    if (!confirmed) {
      setStatus("Bulk cleanup cancelled.");
      return;
    }

    await deleteManagedArtifacts(selectedTargets.map((entry) => entry.path), kind === "all" ? "Filtered artifacts" : kind === "scratch" ? "Filtered scratch paths" : "Filtered output files");
  }

  async function clearScratchPool(path: string, label: string): Promise<void> {
    await deleteManagedArtifact(path, label, () => {
      if (result?.workDir === path) {
        setWorkDirStats(null);
      }
      if (source?.path === path) {
        setSourcePathStats(null);
      }
      if (result?.outputPath === path || outputPath === path) {
        setOutputPathStats(null);
      }
    });
  }

  async function saveRating(nextRating: number | null): Promise<void> {
    try {
      setIsSavingRating(true);
      setError(null);
      const config = await desktopApi.saveModelRating(modelId, nextRating);
      setAppConfig(config);
      setStatus(nextRating === null ? "Model rating cleared." : `Saved ${selectedModel.label} rating.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to save model rating.");
    } finally {
      setIsSavingRating(false);
    }
  }

  async function runBlindComparison(): Promise<void> {
    if (!source) {
      setError("Select a source video before starting blind comparison.");
      return;
    }

    if (blindComparisonCandidates.length < 2) {
      setError("Blind comparison needs at least two runnable models.");
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await ensureRuntime();
      const duration = previewDurationSeconds ?? 8;
      const shuffledModels = shuffleModels(blindComparisonCandidates.map((candidate) => candidate.value));
      const runToken = Date.now().toString(36);
      const startedEntries = shuffledModels.map((candidateModelId, index) => {
        const anonymousLabel = `Sample ${String.fromCharCode(65 + index)}`;
        return {
          sampleId: `sample-${index + 1}`,
          anonymousLabel,
          modelId: candidateModelId,
          jobId: null,
          status: createPendingComparisonJob(),
        };
      });

      setBlindComparison({
        state: "running",
        entries: startedEntries,
        previewDurationSeconds: duration,
        selectedSampleId: null,
        winnerModelId: null,
        revealed: false,
        error: null,
      });
      setStatus("Blind comparison queued.");

      void (async () => {
        for (const entry of startedEntries) {
          try {
            const outputPath = blindComparisonOutputPath(source, container, entry.modelId, entry.anonymousLabel, runToken);
            const jobId = await desktopApi.startPipeline(
              buildPipelineRequest(entry.modelId, outputPath, true, duration)
            );

            setBlindComparison((current) => current ? {
              ...current,
              entries: current.entries.map((candidate) => candidate.sampleId === entry.sampleId
                ? { ...candidate, jobId, status: createQueuedJob(jobId) }
                : candidate),
            } : current);

            while (true) {
              const nextStatus = await desktopApi.getPipelineJob(jobId);
              setBlindComparison((current) => current ? {
                ...current,
                entries: current.entries.map((candidate) => candidate.sampleId === entry.sampleId
                  ? { ...candidate, status: nextStatus }
                  : candidate),
              } : current);

              if (nextStatus.state === "failed") {
                const nextError = nextStatus.error ?? nextStatus.progress.message;
                setBlindComparison((current) => current ? {
                  ...current,
                  state: "failed",
                  error: nextError,
                } : current);
                setError(nextError);
                setStatus("Blind comparison failed.");
                return;
              }

              if (nextStatus.state === "succeeded") {
                break;
              }

              await delay(1000);
            }
          } catch (caught) {
            const nextError = caught instanceof Error ? caught.message : String(caught);
            setBlindComparison((current) => current ? {
              ...current,
              state: "failed",
              error: nextError,
            } : current);
            setError(nextError);
            setStatus("Blind comparison failed.");
            return;
          }
        }

        setBlindComparison((current) => current ? {
          ...current,
          state: "ready",
        } : current);
        setStatus("Blind comparison ready.");
      })();
    } catch (caught) {
      setBlindComparison(null);
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Blind comparison failed.");
    } finally {
      setIsBusy(false);
    }
  }

  async function chooseBlindWinner(sampleId: string): Promise<void> {
    if (!blindComparison || !source) {
      return;
    }

    const winner = blindComparison.entries.find((entry) => entry.sampleId === sampleId);
    if (!winner) {
      return;
    }

    try {
      setError(null);
      const config = await desktopApi.recordBlindComparisonSelection({
        sourcePath: source.path,
        previewDurationSeconds: blindComparison.previewDurationSeconds,
        winnerModelId: winner.modelId,
        candidateModelIds: blindComparison.entries.map((entry) => entry.modelId),
      });
      setAppConfig(config);
      setBlindComparison((current) => current ? {
        ...current,
        selectedSampleId: sampleId,
        winnerModelId: winner.modelId,
        revealed: true,
      } : current);
      setStatus(`Blind comparison winner saved from ${winner.anonymousLabel}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to save blind comparison result.");
    }
  }

  function updateContainer(nextContainer: OutputContainer): void {
    setContainer(nextContainer);
    setOutputPath((currentPath) => {
      if (!currentPath) {
        return currentPath;
      }

      const withoutKnownExtension = currentPath.replace(/\.(mp4|mkv)$/i, "");
      return normalizeOutputPath(withoutKnownExtension, nextContainer);
    });
  }

  function getLargestCropRect(): NormalizedCropRect | null {
    if (!source || outputMode !== "cropTo4k") {
      return null;
    }

    return defaultCropRect(
      { width: source.width, height: source.height },
      {
        ...sizingOptions,
        cropLeft: null,
        cropTop: null,
        cropWidth: null,
        cropHeight: null,
      }
    );
  }

  function beginCropDrag(handle: CropHandle, event: ReactMouseEvent<HTMLElement>): void {
    const interactiveCropRect = cropRect ?? getLargestCropRect();

    if (!interactiveCropRect) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    setDragState({
      handle,
      startX: event.clientX,
      startY: event.clientY,
      startRect: interactiveCropRect
    });
  }

  function nudgeCrop(deltaLeft: number, deltaTop: number): void {
    const interactiveCropRect = cropRect ?? getLargestCropRect();

    if (!interactiveCropRect) {
      return;
    }

    setCropRect(offsetCropRect(interactiveCropRect, deltaLeft, deltaTop));
  }

  function maximizeCrop(): void {
    const largestCropRect = getLargestCropRect();
    if (!largestCropRect) {
      return;
    }

    setCropRect(largestCropRect);
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Windows-first video upscaler evaluation</p>
          <h1>Upscaler</h1>
          <p className="summary">
            Compare Real-ESRGAN-first outputs, inspect 4K framing behavior, rate models from the
            shared catalog, and run blind sample comparisons before committing to a full export.
          </p>
        </div>
        <div className="status-card">
          <span className="status-label">MVP Track</span>
          <strong>{selectedModel.label}</strong>
          <span>{progressMessage}</span>
          <div className="progress-shell" aria-label="Pipeline progress">
            <div className="progress-bar" style={{ width: `${progressPercent}%` }} />
          </div>
          <span>{progressPercent}%</span>
        </div>
      </section>

      <section className="workspace-grid primary-panel-grid">
        <ExpandablePanel
          title="Input"
          subtitle={source ? `${source.width} x ${source.height} • ${source.container.toUpperCase()}` : "Load a source video"}
          isOpen={isInputPanelOpen}
          onToggle={() => setIsInputPanelOpen((current) => !current)}
          testId="input-panel"
        >
          <div className="source-panel">
            <button data-testid="select-video-button" className="action-button" onClick={() => void selectVideo()} disabled={isBusy || isPipelineRunning || isBlindComparisonRunning || isSourceConversionRunning}>
              Select Video
            </button>
            {previewSrc ? (
              <div
                className={`preview-shell interactive-preview${outputMode === "cropTo4k" && isCropEditing ? " crop-enabled" : ""}`}
                ref={previewRef}
                style={source ? { aspectRatio: `${source.width} / ${source.height}` } : undefined}
              >
                <video
                  key={previewSrc ?? 'no-preview-src'}
                  ref={sourcePreviewVideoRef}
                  data-testid="source-preview"
                  className="source-preview"
                  controls
                  preload="metadata"
                  onLoadedMetadata={handleSourcePreviewLoadedMetadata}
                  onPause={() => setSourcePreviewPlaying(false)}
                  onPlay={() => setSourcePreviewPlaying(true)}
                  onEnded={() => setSourcePreviewPlaying(false)}
                  onTimeUpdate={handleSourcePreviewTimeUpdate}
                >
                  {previewSrc ? <source src={previewSrc} type={sourcePreviewMimeType} /> : null}
                </video>
                <div className="source-preview-toolbar" data-testid="source-preview-toolbar">
                  <div className="source-preview-transport-row">
                    <button
                      type="button"
                      className="source-preview-toolbar-button"
                      data-testid="source-preview-play-toggle"
                      onClick={() => void toggleSourcePreviewPlayback()}
                    >
                      {sourcePreviewPlaying ? "Pause" : "Play"}
                    </button>
                    <button
                      type="button"
                      className="source-preview-toolbar-button source-preview-toolbar-button-secondary"
                      data-testid="source-preview-restart"
                      onClick={restartSourcePreviewPlayback}
                    >
                      Restart
                    </button>
                    <span className="source-preview-timecode" data-testid="source-preview-timecode">
                      {formatClockTime(sourcePreviewCurrentTime)} / {formatClockTime(sourcePreviewDuration || source?.durationSeconds || 0)}
                    </span>
                  </div>
                  <input
                    data-testid="source-preview-seek"
                    className="source-preview-seek"
                    type="range"
                    min={0}
                    max={sourcePreviewSeekMax}
                    step={0.01}
                    value={Math.min(sourcePreviewCurrentTime, sourcePreviewSeekMax)}
                    onInput={(event) => seekSourcePreview(Number((event.target as HTMLInputElement).value))}
                    onChange={(event) => seekSourcePreview(Number(event.target.value))}
                  />
                </div>
                {cropOverlayStyle ? (
                  <div
                    data-testid="crop-overlay"
                    className={`crop-overlay${isCropEditing ? " crop-overlay-editing" : ""}`}
                    style={cropOverlayStyle}
                    onMouseDown={isCropEditing ? (event) => beginCropDrag("move", event) : undefined}
                  >
                    <span className="crop-overlay-label" data-testid="crop-overlay-label">
                      {isCropEditing ? "Crop Editing" : "Crop Frame"}
                    </span>
                    {isCropEditing ? (
                      <>
                        <button type="button" data-testid="crop-handle-nw" className="crop-handle handle-nw" onMouseDown={(event) => beginCropDrag("nw", event)} aria-label="Resize crop from top left" />
                        <button type="button" data-testid="crop-handle-ne" className="crop-handle handle-ne" onMouseDown={(event) => beginCropDrag("ne", event)} aria-label="Resize crop from top right" />
                        <button type="button" data-testid="crop-handle-sw" className="crop-handle handle-sw" onMouseDown={(event) => beginCropDrag("sw", event)} aria-label="Resize crop from bottom left" />
                        <button type="button" data-testid="crop-handle-se" className="crop-handle handle-se" onMouseDown={(event) => beginCropDrag("se", event)} aria-label="Resize crop from bottom right" />
                        <button type="button" data-testid="crop-move-handle" className="crop-move-handle" onMouseDown={(event) => beginCropDrag("move", event)} aria-label="Move crop selection">
                          Move Crop
                        </button>
                      </>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="preview-shell preview-placeholder">
                <span>Input preview appears here after you select a video.</span>
              </div>
            )}
            {source ? (
              <div className="preview-framing-controls" data-testid="preview-framing-controls">
                <label>
                  Preview Framing Aspect
                  <select data-testid="aspect-ratio-select" value={aspectRatioPreset} onChange={(event) => setAspectRatioPreset(event.target.value as AspectRatioPreset)}>
                    {aspectRatioPresets.map((preset) => (
                      <option key={preset.value} value={preset.value}>{preset.label}</option>
                    ))}
                  </select>
                </label>
                {aspectRatioPreset === "custom" ? (
                  <div className="preview-framing-custom-grid">
                    <label>
                      Custom Width
                      <input data-testid="custom-aspect-width-input" type="number" min={1} step={1} value={customAspectWidthInput} onChange={(event) => setCustomAspectWidthInput(event.target.value)} />
                    </label>
                    <label>
                      Custom Height
                      <input data-testid="custom-aspect-height-input" type="number" min={1} step={1} value={customAspectHeightInput} onChange={(event) => setCustomAspectHeightInput(event.target.value)} />
                    </label>
                  </div>
                ) : null}
                <p className="summary preview-framing-summary" data-testid="preview-framing-summary">
                  {outputMode === "cropTo4k"
                    ? `The crop box previews a ${aspectRatioValue.toFixed(3)}:1 target framing inside the original source frame.`
                    : `Target framing is currently ${aspectRatioValue.toFixed(3)}:1. Switch to Crop To Fill Target to reposition the preview crop.`}
                </p>
              </div>
            ) : null}
            {source ? (
              <div className="source-preview-actions">
                <button type="button" className="action-button secondary-button" data-testid="open-source-external-preview" onClick={() => void openMediaInDefaultApp(source.path)}>
                  Open Source Externally
                </button>
                {!supportsEmbeddedFullLengthPreview(source.container) ? (
                  <button
                    type="button"
                    className="action-button secondary-button"
                    data-testid="convert-source-to-mp4-button"
                    onClick={() => void convertCurrentSourceToMp4()}
                    disabled={isBusy || isPipelineRunning || isBlindComparisonRunning || isSourceConversionRunning}
                  >
                    Fast Convert To MP4
                  </button>
                ) : null}
              </div>
            ) : null}
            {source && outputMode === "cropTo4k" && cropOverlayStyle ? (
              <div className="crop-tools-panel" data-testid="crop-tools-panel">
                <div className="crop-tools-actions">
                  <button type="button" className="action-button secondary-button" data-testid="toggle-crop-edit-button" onClick={() => setIsCropEditing((current) => !current)}>
                    {isCropEditing ? "Done Editing Crop" : "Edit Crop"}
                  </button>
                  <button type="button" className="action-button secondary-button" data-testid="maximize-crop-button" onClick={maximizeCrop}>
                    Maximize Crop
                  </button>
                </div>
                <div className="crop-nudge-controls" data-testid="crop-nudge-controls">
                <button type="button" className="action-button secondary-button crop-nudge-button crop-nudge-up" data-testid="crop-nudge-up" onClick={() => nudgeCrop(0, -0.02)} disabled={!isCropEditing}>
                  Up
                </button>
                <button type="button" className="action-button secondary-button crop-nudge-button crop-nudge-left" data-testid="crop-nudge-left" onClick={() => nudgeCrop(-0.02, 0)} disabled={!isCropEditing}>
                  Left
                </button>
                <span className="crop-nudge-label">{isCropEditing ? "Nudge Crop" : "Enable Edit"}</span>
                <button type="button" className="action-button secondary-button crop-nudge-button crop-nudge-right" data-testid="crop-nudge-right" onClick={() => nudgeCrop(0.02, 0)} disabled={!isCropEditing}>
                  Right
                </button>
                <button type="button" className="action-button secondary-button crop-nudge-button crop-nudge-down" data-testid="crop-nudge-down" onClick={() => nudgeCrop(0, 0.02)} disabled={!isCropEditing}>
                  Down
                </button>
                </div>
              </div>
            ) : null}
            {source ? (
              <dl className="facts">
                <div><dt>Path</dt><dd>{source.path}</dd></div>
                <div><dt>Preview</dt><dd data-testid="source-preview-mode">{previewUpgradeAvailable ? "Full-length converted preview" : usingFallbackPreviewClip ? "Short fallback preview clip" : "Direct source playback"}</dd></div>
                <div><dt>Input Size</dt><dd>{formatBytes(sourcePathStats?.sizeBytes ?? 0)}</dd></div>
                <div><dt>Resolution</dt><dd>{source.width} x {source.height}</dd></div>
                <div><dt>Duration</dt><dd>{source.durationSeconds.toFixed(2)}s</dd></div>
                <div><dt>Frame Rate</dt><dd>{source.frameRate.toFixed(3)} fps</dd></div>
                <div><dt>Audio</dt><dd>{source.hasAudio ? "Present" : "Missing"}</dd></div>
                <div><dt>Container</dt><dd>{source.container}</dd></div>
              </dl>
            ) : null}
            {source && !supportsEmbeddedFullLengthPreview(source.container) ? (
              <p className="summary" data-testid="source-preview-guidance">
                {previewUpgradeAvailable
                  ? "The embedded player is now using a full-length converted preview. Use Fast Convert To MP4 if you also want the actual working source replaced with that MP4 before upscaling."
                  : previewUpgradePending
                    ? "This container starts on a short fallback clip so playback is available immediately. A full-length playable preview is being prepared in the background."
                    : canAutoUpgradePreview
                      ? "This container uses a short fallback clip in the embedded player. Use Fast Convert To MP4 to make a full-length working copy before running the upscale."
                      : "This source is using a lightweight fallback preview clip to keep the desktop preview responsive. Use Fast Convert To MP4 if you want a longer playable working copy before running the upscale."}
              </p>
            ) : (
              !source ? <p className="summary">Select a local video file to probe and run. AVI sources can be fast-converted to MP4 when needed.</p> : null
            )}
          </div>
        </ExpandablePanel>

        <ExpandablePanel
          title="Output"
          subtitle={result ? "Output ready" : isPipelineRunning ? "Encoding in progress" : "Configure and run"}
          isOpen={isOutputPanelOpen}
          onToggle={() => setIsOutputPanelOpen((current) => !current)}
          testId="output-panel"
        >
          <label>
            Model
            <select data-testid="model-select" value={modelId} onChange={(event) => setModelId(event.target.value as ModelId)}>
              <optgroup label="Available Now">
                {runnableModels.map((model) => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Planned">
                {plannedModels.map((model) => (
                  <option key={model.value} value={model.value} disabled>
                    {model.label} (not implemented)
                  </option>
                ))}
              </optgroup>
            </select>
          </label>
          <section className="catalog-card" data-testid="model-details-card">
            <div className="catalog-card-header">
              <strong data-testid="selected-model-label">{selectedModel.label}</strong>
              <span className={`catalog-chip execution-${selectedModel.executionStatus}`} data-testid="selected-model-status">
                {isSelectedModelImplemented ? selectedModel.executionStatus : "not implemented"}
              </span>
            </div>
            <p className="summary" data-testid="selected-model-summary">{selectedModel.summary}</p>
            {!isSelectedModelImplemented ? (
              <p className="summary" data-testid="selected-model-availability">
                This model is visible in the catalog but is not implemented yet, so it cannot be selected for export.
              </p>
            ) : null}
            <dl className="facts compact-facts">
              <div><dt>Backend</dt><dd>{selectedBackend.label}</dd></div>
              <div><dt>Loader</dt><dd>{selectedModel.loader}</dd></div>
              <div><dt>Runtime Model</dt><dd>{selectedModel.runtimeModelName}</dd></div>
              <div><dt>Support Tier</dt><dd>{selectedModel.supportTier}</dd></div>
              <div><dt>Quality Rank</dt><dd>#{selectedModel.qualityRank}</dd></div>
              <div><dt>Native Scale</dt><dd>{selectedModel.nativeScale}x</dd></div>
              <div><dt>Video Path</dt><dd>{selectedModel.videoNative ? "Native video pipeline" : "Frame-by-frame pipeline"}</dd></div>
              <div><dt>Suitability</dt><dd>{selectedModel.mediaSuitability.join(", ")}</dd></div>
              <div><dt>GPU Routing</dt><dd>{selectedModel.specialHandling.supportsGpuId ? "Explicit GPU id supported" : "Automatic device selection only"}</dd></div>
            </dl>
          </section>
          <label>
            Rating
            <select
              data-testid="model-rating-select"
              value={selectedModelRating !== null ? String(selectedModelRating) : ""}
              onChange={(event) => void saveRating(event.target.value === "" ? null : Number(event.target.value))}
              disabled={isSavingRating}
            >
              <option value="">No saved rating</option>
              <option value="1">1 / 5</option>
              <option value="2">2 / 5</option>
              <option value="3">3 / 5</option>
              <option value="4">4 / 5</option>
              <option value="5">5 / 5</option>
            </select>
          </label>
          <p className="summary" data-testid="rating-summary">
            {selectedModelRating !== null
              ? `Saved rating: ${selectedModelRating}/5. Persisted in config/model_preferences.json.`
              : "No saved rating yet. Ratings persist in config/model_preferences.json."}
          </p>
          <label>
            GPU Device
            <select
              data-testid="gpu-select"
              value={selectedGpuId !== null ? String(selectedGpuId) : ""}
              onChange={(event) => setSelectedGpuId(event.target.value === "" ? null : Number(event.target.value))}
              disabled={!runtime || runtime.availableGpus.length === 0}
            >
              {!runtime ? <option value="">Prepare runtime first</option> : null}
              {runtime && runtime.availableGpus.length === 0 ? <option value="">No Vulkan GPUs detected</option> : null}
              {runtime?.availableGpus.map((gpu) => (
                <option key={gpu.id} value={gpu.id}>{gpu.id}: {gpu.name} ({gpu.kind})</option>
              ))}
            </select>
          </label>
          {runtime ? (
            <p className="summary">
              {selectedGpu
                ? `Using NCNN/Vulkan GPU ${selectedGpu.id}: ${selectedGpu.name}.`
                : "Runtime detected no explicit Vulkan GPU selection."}
            </p>
          ) : null}
          <label>
            Output Mode
            <select data-testid="output-mode-select" value={outputMode} onChange={(event) => setOutputMode(event.target.value as OutputMode)}>
              {outputModes.map((mode) => (
                <option key={mode.value} value={mode.value}>{mode.label}</option>
              ))}
            </select>
          </label>
          <label>
            Resolution Basis
            <select data-testid="resolution-basis-select" value={resolutionBasis} onChange={(event) => setResolutionBasis(event.target.value as ResolutionBasis)}>
              {resolutionBases.map((basis) => (
                <option key={basis.value} value={basis.value}>{basis.label}</option>
              ))}
            </select>
          </label>
          <div className="dual-field-grid">
            <label>
              Target Width
              <input
                data-testid="target-width-input"
                type="number"
                min={2}
                step={2}
                value={displayedWidth}
                onChange={(event) => setTargetWidthInput(event.target.value)}
                readOnly={resolutionBasis === "height"}
              />
            </label>
            <label>
              Target Height
              <input
                data-testid="target-height-input"
                type="number"
                min={2}
                step={2}
                value={displayedHeight}
                onChange={(event) => setTargetHeightInput(event.target.value)}
                readOnly={resolutionBasis === "width"}
              />
            </label>
          </div>
          {outputMode === "cropTo4k" ? <p className="summary">Use Edit Crop to move the framing box without blocking the source player controls. The derived dimension is computed from the selected resolution basis and stays editable through the driving axis.</p> : null}
          <label>
            Quality Preset
            <select
              data-testid="quality-preset-select"
              value={qualityPreset}
              onChange={(event) => setQualityPreset(event.target.value as QualityPreset)}
            >
              {qualityPresets.map((preset) => (
                <option key={preset.value} value={preset.value}>{preset.label}</option>
              ))}
            </select>
          </label>
          {supportsPytorchRunner ? (
            <label>
              PyTorch Runner
              <select data-testid="pytorch-runner-select" value={pytorchRunner} onChange={(event) => setPytorchRunner(event.target.value as PytorchRunner)}>
                {pytorchRunners.map((runner) => (
                  <option key={runner.value} value={runner.value}>{runner.label}</option>
                ))}
              </select>
              <span className="summary">
                TensorRT builds a cached engine on the first run for supported PyTorch image SR models, then reuses it on later runs.
              </span>
            </label>
          ) : null}
          <label className="checkbox-row">
            <input data-testid="preview-mode-checkbox" type="checkbox" checked={previewMode} onChange={(event) => setPreviewMode(event.target.checked)} />
            <span>Quick Test Mode</span>
          </label>
          <label>
            Preview Duration Seconds
            <input
              data-testid="preview-duration-input"
              type="number"
              min={1}
              max={30}
              step={1}
              value={previewDurationInput}
              onChange={(event) => setPreviewDurationInput(event.target.value)}
              disabled={!previewMode}
            />
          </label>
          <label>
            Export Chunk Seconds
            <input
              data-testid="segment-duration-input"
              type="number"
              min={1}
              max={120}
              step={1}
              value={segmentDurationInput}
              onChange={(event) => setSegmentDurationInput(event.target.value)}
              disabled={previewMode}
            />
            <span className="summary">Full exports buffer this many seconds per restartable chunk. Larger chunks reduce intermediate MKV overhead but reduce restart granularity. Quick Test always runs as one segment.</span>
          </label>
          <div className="dual-field-grid">
            <label>
              Codec
              <select data-testid="codec-select" value={codec} onChange={(event) => setCodec(event.target.value as VideoCodec)}>
                {codecs.map((entry) => (
                  <option key={entry.value} value={entry.value}>{entry.label}</option>
                ))}
              </select>
            </label>
            <label>
              Container
              <select data-testid="container-select" value={container} onChange={(event) => updateContainer(event.target.value as OutputContainer)}>
                {containers.map((entry) => (
                  <option key={entry.value} value={entry.value}>{entry.label}</option>
                ))}
              </select>
            </label>
          </div>
          <label>
            Tile Size
            <input data-testid="tile-size-input" type="number" min={0} step={32} value={tileSize} onChange={(event) => setTileSize(Number(event.target.value))} />
            <span className="summary">Use 0 for auto: PyTorch defaults to 384 on balanced quality, NCNN defaults to 256.</span>
          </label>
          <label>
            CRF
            <input data-testid="crf-input" type="number" min={0} max={51} value={crf} onChange={(event) => setCrf(Number(event.target.value))} />
          </label>
          <label>
            Output File
            <div className="path-picker-row">
              <input data-testid="output-path-input" className="path-readonly-input" type="text" value={outputPath ?? defaultOutputPath(source, container, modelId)} readOnly />
              <button data-testid="save-output-button" className="action-button secondary-button" onClick={() => void chooseOutputFile()} disabled={isRunDisabled}>
                Save As
              </button>
            </div>
          </label>
          {framing ? (
            <div className="framing-preview">
              <span>Target Canvas</span>
              <strong>{framing.canvas.width} x {framing.canvas.height}</strong>
              <span>Aspect Ratio</span>
              <strong>{framing.aspectRatio.toFixed(3)} : 1</strong>
              <span>Scaled Source</span>
              <strong>{framing.scaled.width} x {framing.scaled.height}</strong>
              <span>Selected Input Window</span>
              <strong>{framing.cropWindow.width} x {framing.cropWindow.height} at {framing.cropWindow.offsetX}, {framing.cropWindow.offsetY}</strong>
            </div>
          ) : null}
          <button data-testid="run-upscale-button" className="action-button" onClick={() => void runPipeline()} disabled={isRunDisabled}>
            {isPipelineRunning ? "Upscaling..." : "Run Upscale"}
          </button>
          {!isSelectedModelImplemented ? (
            <p className="summary" data-testid="run-disabled-reason">
              {selectedModel.label} is not implemented yet, so export is disabled.
            </p>
          ) : null}
          {runtime ? (
            <dl className="facts compact-facts">
              <div><dt>FFmpeg</dt><dd>{runtime.ffmpegPath}</dd></div>
              <div><dt>Real-ESRGAN</dt><dd>{runtime.realesrganPath}</dd></div>
              <div><dt>Selected Model</dt><dd>{selectedModel.label}</dd></div>
              <div><dt>Detected GPUs</dt><dd>{runtime.availableGpus.length > 0 ? runtime.availableGpus.map((gpu) => `${gpu.id}: ${gpu.name}`).join(" | ") : "None detected"}</dd></div>
              <div><dt>Selected GPU</dt><dd>{selectedGpu ? `${selectedGpu.id}: ${selectedGpu.name}` : "Automatic / none"}</dd></div>
              <div><dt>Saved Ratings</dt><dd>{Object.keys(appConfig?.modelRatings ?? {}).length}</dd></div>
              <div><dt>Blind Picks Logged</dt><dd>{appConfig?.blindComparisons.length ?? 0}</dd></div>
            </dl>
          ) : (
            <p className="summary">Runtime assets download on first use.</p>
          )}
          {sourceConversionJob && !pipelineJob ? (
            <div className="job-progress-panel" data-testid="conversion-progress-panel">
              <div className="catalog-card-header">
                <strong>Source Conversion</strong>
                <span>{sourceConversionJob.progress.percent}%</span>
              </div>
              <div className="progress-shell large-progress">
                <div className="progress-bar" style={{ width: `${sourceConversionJob.progress.percent}%` }} />
              </div>
              <strong data-testid="conversion-progress-message">{sourceConversionJob.progress.message}</strong>
              <span>{sourceConversionJob.progress.phase}</span>
              <div className="progress-stat-grid">
                <span data-testid="conversion-progress-current">Processed: {formatDurationProgress(sourceConversionJob.progress.processedFrames)}</span>
                <span data-testid="conversion-progress-total">Total: {formatDurationProgress(sourceConversionJob.progress.totalFrames || 0)}</span>
              </div>
            </div>
          ) : null}
          {pipelineJob ? (
            <div className="job-progress-panel" data-testid="job-progress-panel">
              <div className="catalog-card-header">
                <strong>Upscale Progress</strong>
                <span>{pipelineJob.progress.percent}%</span>
              </div>
              <div className="progress-shell large-progress">
                <div className="progress-bar" style={{ width: `${pipelineJob.progress.percent}%` }} />
              </div>
              <strong data-testid="progress-message">{pipelineJob.progress.message}</strong>
              <span>{pipelineJob.progress.phase}</span>
              <div className="progress-live-summary" data-testid="progress-live-summary">
                <strong data-testid="progress-current-activity">{pipelineActivityTitle}</strong>
                <span data-testid="progress-current-detail">{pipelineActivityDetail}</span>
                <span data-testid="progress-last-update">Last update {pipelineLastUpdateLabel}</span>
              </div>
              {pipelineJob.progress.segmentIndex && pipelineJob.progress.segmentCount ? (
                <div className="progress-stat-grid">
                  <span data-testid="progress-segment-counter">Chunk: {pipelineJob.progress.segmentIndex}/{pipelineJob.progress.segmentCount}</span>
                  <span data-testid="progress-segment-frames">Chunk Frames: {pipelineJob.progress.segmentProcessedFrames ?? 0}/{pipelineJob.progress.segmentTotalFrames ?? 0}</span>
                  {pipelineJob.progress.batchCount ? <span data-testid="progress-batch-counter">Batch: {pipelineJob.progress.batchIndex ?? 0}/{pipelineJob.progress.batchCount}</span> : null}
                </div>
              ) : null}
              <div className="phase-progress-grid">
                {pipelinePhaseBars.map((phaseBar) => (
                  <div key={phaseBar.id} className="phase-progress-card" data-testid={`phase-progress-${phaseBar.id}`}>
                    <div className="phase-progress-header">
                      <strong>{phaseBar.label}</strong>
                      <span>{phaseBar.summary}</span>
                    </div>
                    <div className="progress-shell phase-progress-shell">
                      <div className="progress-bar" style={{ width: `${phaseBar.value * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
              <div className="progress-stat-grid">
                <span data-testid="progress-total-frames">Total Frames: {pipelineJob.progress.totalFrames || "?"}</span>
                <span data-testid="progress-extracted-frames">Extracted PNGs: {pipelineJob.progress.extractedFrames}</span>
                <span data-testid="progress-upscaled-frames">Upscaled PNGs: {pipelineJob.progress.upscaledFrames}</span>
                <span data-testid="progress-encoded-frames">Encoded Frames: {pipelineJob.progress.encodedFrames}</span>
                <span data-testid="progress-remuxed-frames">Audio Remux Frames: {pipelineJob.progress.remuxedFrames}</span>
              </div>
              <div className="progress-stat-grid">
                <span data-testid="progress-average-fps">Average Throughput: {formatFramesPerSecond(pipelineJob.progress.averageFramesPerSecond)}</span>
                <span data-testid="progress-rolling-fps">Current Throughput: {formatFramesPerSecond(pipelineJob.progress.rollingFramesPerSecond)}</span>
                <span data-testid="progress-eta">ETA: {formatElapsedSeconds(pipelineJob.progress.estimatedRemainingSeconds)}</span>
                <span data-testid="progress-elapsed">Elapsed: {formatElapsedSeconds(pipelineJob.progress.elapsedSeconds)}</span>
                <span data-testid="progress-process-rss">Worker RAM: {formatBytes(pipelineJob.progress.processRssBytes ?? 0)}</span>
              </div>
              <div className="progress-stat-grid">
                <span data-testid="progress-gpu-memory">GPU Memory: {formatGpuMemory(pipelineJob.progress.gpuMemoryUsedBytes, pipelineJob.progress.gpuMemoryTotalBytes)}</span>
                <span data-testid="workdir-size">Job Scratch Size: {formatBytes(progressScratchSizeBytes)}</span>
                <span data-testid="output-file-size">Output Size: {formatBytes(progressOutputSizeBytes)}</span>
                <span data-testid="progress-stage-timings">Stage Times: {formatStageTimings(pipelineJob.progress)}</span>
              </div>
              <div className="progress-event-log" data-testid="progress-event-log">
                {pipelineProgressEvents.length > 0 ? [...pipelineProgressEvents].reverse().map((entry) => (
                  <div key={`${entry.key}-${entry.timestamp}`} className="progress-event-row">
                    <div className="progress-event-header">
                      <strong>{entry.title}</strong>
                      <span>{entry.percent}%</span>
                    </div>
                    <span>{entry.detail}</span>
                    <span className="progress-event-timestamp">{formatExactTimestamp(entry.timestamp)}</span>
                  </div>
                )) : (
                  <div className="progress-event-row">
                    <strong>Waiting for live worker updates</strong>
                    <span>The panel will add milestone entries as the pipeline advances.</span>
                  </div>
                )}
              </div>
            </div>
          ) : null}
          {result ? (
            <>
              {source && resultPreviewSrc ? (
                <div className="comparison-grid">
                  <div className="comparison-card">
                    <span>Source Preview</span>
                    <button type="button" className="preview-launcher" data-testid="source-result-open-button" onClick={() => void openMediaInDefaultApp(source.path)}>
                      <video data-testid="source-result-preview" className="result-preview clickable-preview" preload="metadata" src={previewSrc ?? undefined} muted />
                      <span className="preview-launch-hint">Click to open in the default video app</span>
                    </button>
                  </div>
                  <div className="comparison-card">
                    <span>Output Preview</span>
                    <button type="button" className="preview-launcher" data-testid="result-open-button" onClick={() => void openMediaInDefaultApp(result.outputPath)}>
                      <video data-testid="result-preview" className="result-preview clickable-preview" preload="metadata" src={resultPreviewSrc} muted />
                      <span className="preview-launch-hint">Click to open in the default video app</span>
                    </button>
                  </div>
                </div>
              ) : null}
              <dl className="facts compact-facts">
                <div><dt>Output</dt><dd data-testid="result-output-path">{result.outputPath}</dd></div>
                <div><dt>Output Size</dt><dd>{formatBytes(outputPathStats?.sizeBytes ?? 0)}</dd></div>
                <div><dt>Work Dir</dt><dd>{result.workDir}</dd></div>
                <div><dt>Scratch Size</dt><dd>{formatBytes(workDirStats?.sizeBytes ?? 0)}</dd></div>
                <div><dt>Frames</dt><dd>{result.frameCount}</dd></div>
                <div><dt>Codec</dt><dd>{result.codec}</dd></div>
                <div><dt>Container</dt><dd>{result.container}</dd></div>
                <div><dt>Audio Sync</dt><dd>{result.hadAudio ? "Original audio remuxed" : "No source audio"}</dd></div>
              </dl>
              <div data-testid="pipeline-log" className="log-box">
                {result.log.map((line, index) => (
                  <div key={`${index}-${line.slice(0, 12)}`}>{line}</div>
                ))}
              </div>
            </>
          ) : (
            <ul className="simple-list">
              <li>Select a source file and verify its preview.</li>
              <li>Pick the model, codec, container, and output file.</li>
              <li>Use Quick Test Mode or blind comparison before a full export.</li>
            </ul>
          )}
        </ExpandablePanel>
      </section>

      <section className="advanced-panel-stack">
        <ExpandablePanel
          title="Blind Test"
          subtitle={`${blindComparisonCandidates.length} runnable models`}
          isOpen={isBlindPanelOpen}
          onToggle={() => setIsBlindPanelOpen((current) => !current)}
          testId="blind-test-panel"
        >
          <section className="blind-comparison-panel" data-testid="blind-comparison-panel">
            <p className="summary">
              Runs anonymized {previewDurationSeconds ?? 8}s preview exports for every runnable model,
              then lets you pick the best sample before the app reveals which model produced it.
            </p>
            <button
              data-testid="run-blind-comparison-button"
              className="action-button secondary-button"
              onClick={() => void runBlindComparison()}
              disabled={isBlindComparisonDisabled}
            >
              {isBlindComparisonRunning ? "Blind Comparison Running..." : "Run Blind Comparison"}
            </button>
            {blindComparison ? (
              <div className="blind-sample-grid">
                {blindComparison.entries.map((entry) => {
                  const actualModel = getModelDefinition(entry.modelId);
                  const actualBackend = getBackendDefinition(actualModel.backendId);
                  const isWinner = blindComparison.selectedSampleId === entry.sampleId;
                  const previewPath = entry.status.result?.outputPath;
                  return (
                    <article key={entry.sampleId} className={`blind-sample-card${isWinner ? " blind-sample-card-selected" : ""}`} data-testid={`blind-sample-${entry.sampleId}`}>
                      <div className="blind-sample-header">
                        <strong>{entry.anonymousLabel}</strong>
                        <span>{entry.status.progress.percent}%</span>
                      </div>
                      <div className="progress-shell sample-progress" aria-label={`${entry.anonymousLabel} progress`}>
                        <div className="progress-bar" style={{ width: `${entry.status.progress.percent}%` }} />
                      </div>
                      {previewPath ? (
                        <button
                          type="button"
                          className="preview-launcher"
                          data-testid={`blind-open-${entry.sampleId}`}
                          onClick={() => void openMediaInDefaultApp(previewPath)}
                        >
                          <video className="result-preview clickable-preview" preload="metadata" src={desktopApi.toPreviewSrc(previewPath)} data-testid={`blind-preview-${entry.sampleId}`} muted />
                          <span className="preview-launch-hint">Click to open in the default video app</span>
                        </button>
                      ) : (
                        <div className="preview-shell preview-placeholder blind-placeholder">
                          <span>{entry.status.progress.message}</span>
                        </div>
                      )}
                      <span className="blind-sample-status">{entry.status.progress.message}</span>
                      {blindComparison.revealed ? (
                        <div className="blind-reveal-block" data-testid={`blind-reveal-${entry.sampleId}`}>
                          <strong>{actualModel.label}</strong>
                          <span>{actualBackend.label}</span>
                          {isWinner ? <span className="winner-pill">Selected winner</span> : null}
                        </div>
                      ) : null}
                      {blindComparison.state === "ready" && !blindComparison.revealed ? (
                        <button className="action-button secondary-button" data-testid={`pick-${entry.sampleId}`} onClick={() => void chooseBlindWinner(entry.sampleId)}>
                          Pick {entry.anonymousLabel}
                        </button>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            ) : null}
            {selectedComparisonEntry && source ? (
              <section className="comparison-inspector" data-testid="comparison-inspector">
                <div className="catalog-card-header">
                  <strong>Comparison Inspector</strong>
                  <span className="catalog-chip">Zoomed source vs sample</span>
                </div>
                <p className="summary">
                  Inspect the source beside one blind sample at larger scale, jump between pixel-focus presets, and open either clip in the default player when you need full controls or a full-size window.
                </p>
                <div className="inspector-sample-row">
                  {comparisonEntries.map((entry) => (
                    <button
                      key={entry.sampleId}
                      type="button"
                      className={`inspector-sample-button${entry.sampleId === selectedComparisonEntry.sampleId ? " inspector-sample-button-active" : ""}`}
                      data-testid={`comparison-select-${entry.sampleId}`}
                      onClick={() => setComparisonSampleId(entry.sampleId)}
                    >
                      {entry.anonymousLabel}
                    </button>
                  ))}
                </div>
                <div className="inspector-controls-grid">
                  <button type="button" className="action-button secondary-button" data-testid="open-source-external-button" onClick={() => void openMediaInDefaultApp(source.path)}>
                    Open Source Externally
                  </button>
                  <button type="button" className="action-button secondary-button" data-testid="open-sample-external-button" onClick={() => void openMediaInDefaultApp(selectedComparisonEntry.status.result?.outputPath ?? "") }>
                    Open Sample Externally
                  </button>
                  <button type="button" className="action-button secondary-button" data-testid="comparison-play-toggle" onClick={() => void toggleComparisonPlayback()}>
                    {comparisonPlaying ? "Pause Comparison" : "Play Comparison"}
                  </button>
                  <button type="button" className="action-button secondary-button" data-testid="comparison-restart-button" onClick={restartComparisonPlayback}>
                    Restart Comparison
                  </button>
                </div>
                <label>
                  Comparison Timeline
                  <input
                    data-testid="comparison-time-slider"
                    type="range"
                    min={0}
                    max={Math.max(comparisonDuration, 0.01)}
                    step={0.01}
                    value={Math.min(comparisonCurrentTime, Math.max(comparisonDuration, 0.01))}
                    onChange={(event) => syncComparisonTime(Number(event.target.value))}
                  />
                </label>
                <div className="comparison-toolbar-grid">
                  <label>
                    Zoom
                    <input data-testid="comparison-zoom-slider" type="range" min={1} max={8} step={0.25} value={comparisonZoom} onChange={(event) => setComparisonZoom(Number(event.target.value))} />
                  </label>
                  <label>
                    Horizontal Focus
                    <input
                      data-testid="comparison-focus-x-slider"
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={comparisonFocusX}
                      onChange={(event) => {
                        setComparisonFocusPresetId("manual");
                        setComparisonFocusX(Number(event.target.value));
                      }}
                    />
                  </label>
                  <label>
                    Vertical Focus
                    <input
                      data-testid="comparison-focus-y-slider"
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={comparisonFocusY}
                      onChange={(event) => {
                        setComparisonFocusPresetId("manual");
                        setComparisonFocusY(Number(event.target.value));
                      }}
                    />
                  </label>
                </div>
                <div className="inspector-focus-row">
                  {comparisonFocusPresets.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      className={`inspector-focus-button${preset.id === comparisonFocusPresetId ? " inspector-focus-button-active" : ""}`}
                      data-testid={`comparison-focus-${preset.id}`}
                      onClick={() => setComparisonFocusPresetId(preset.id)}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
                <p className="summary comparison-hint" data-testid="comparison-focus-hint">
                  {selectedComparisonPreset?.hint ?? "Move around the frame and inspect the sample against the source."}
                </p>
                <div className="comparison-inspector-grid">
                  <article className="comparison-inspector-card">
                    <div className="catalog-card-header">
                      <strong>Source</strong>
                      <span className="catalog-chip">Reference</span>
                    </div>
                    <button type="button" className="inspection-viewport" data-testid="comparison-source-viewport" onClick={() => void openMediaInDefaultApp(source.path)}>
                      <video
                        ref={comparisonSourceVideoRef}
                        className="inspection-video"
                        src={desktopApi.toPreviewSrc(source.previewPath || source.path)}
                        muted
                        playsInline
                        preload="metadata"
                        onLoadedMetadata={handleComparisonLoadedMetadata}
                        onTimeUpdate={handleComparisonSourceTimeUpdate}
                        onPause={() => setComparisonPlaying(false)}
                        onPlay={() => setComparisonPlaying(true)}
                        style={{ transform: `scale(${comparisonZoom})`, transformOrigin: `${comparisonFocusX}% ${comparisonFocusY}%` }}
                      />
                      <span className="inspection-crosshair" />
                    </button>
                  </article>
                  <article className="comparison-inspector-card">
                    <div className="catalog-card-header">
                      <strong>{selectedComparisonEntry.anonymousLabel}</strong>
                      <span className="catalog-chip">Blind sample</span>
                    </div>
                    <button type="button" className="inspection-viewport" data-testid="comparison-sample-viewport" onClick={() => void openMediaInDefaultApp(selectedComparisonEntry.status.result?.outputPath ?? "") }>
                      <video
                        ref={comparisonOutputVideoRef}
                        className="inspection-video"
                        src={desktopApi.toPreviewSrc(selectedComparisonEntry.status.result?.outputPath ?? "")}
                        muted
                        playsInline
                        preload="metadata"
                        onLoadedMetadata={handleComparisonLoadedMetadata}
                        style={{ transform: `scale(${comparisonZoom})`, transformOrigin: `${comparisonFocusX}% ${comparisonFocusY}%` }}
                      />
                      <span className="inspection-crosshair" />
                    </button>
                  </article>
                </div>
              </section>
            ) : null}
            {blindComparison?.error ? <p className="error-text">{blindComparison.error}</p> : null}
          </section>
        </ExpandablePanel>

        <ExpandablePanel
          title="Job Cleanup"
          subtitle={`${cleanupJobs.length} tracked and historical jobs`}
          isOpen={isCleanupPanelOpen}
          onToggle={() => setIsCleanupPanelOpen((current) => !current)}
          testId="job-cleanup-panel"
        >
          <section className="catalog-card cleanup-card" data-testid="cleanup-jobs-card">
            <div className="catalog-card-header">
              <strong>Tracked And Historical Jobs</strong>
              <span className="catalog-chip">Counts from progress JSON</span>
            </div>
            {cleanupJobs.length > 0 ? (
              <>
                <div className="cleanup-filter-row" data-testid="cleanup-filter-row">
                  <button type="button" className={`cleanup-filter-button${cleanupFilter === "all" ? " cleanup-filter-button-active" : ""}`} data-testid="cleanup-filter-all" onClick={() => setCleanupFilter("all")}>
                    All ({cleanupStateCounts.all})
                  </button>
                  <button type="button" className={`cleanup-filter-button${cleanupFilter === "running" ? " cleanup-filter-button-active" : ""}`} data-testid="cleanup-filter-running" onClick={() => setCleanupFilter("running")}>
                    Running ({cleanupStateCounts.running})
                  </button>
                  <button type="button" className={`cleanup-filter-button${cleanupFilter === "succeeded" ? " cleanup-filter-button-active" : ""}`} data-testid="cleanup-filter-succeeded" onClick={() => setCleanupFilter("succeeded")}>
                    Completed ({cleanupStateCounts.succeeded})
                  </button>
                  <button type="button" className={`cleanup-filter-button${cleanupFilter === "cancelled" ? " cleanup-filter-button-active" : ""}`} data-testid="cleanup-filter-cancelled" onClick={() => setCleanupFilter("cancelled")}>
                    Cancelled ({cleanupStateCounts.cancelled})
                  </button>
                  <button type="button" className={`cleanup-filter-button${cleanupFilter === "failed" ? " cleanup-filter-button-active" : ""}`} data-testid="cleanup-filter-failed" onClick={() => setCleanupFilter("failed")}>
                    Failed ({cleanupStateCounts.failed})
                  </button>
                </div>
                <div className="cleanup-toolbar-grid">
                  <label>
                    Search Jobs And Files
                    <input
                      data-testid="cleanup-search-input"
                      type="search"
                      value={cleanupSearch}
                      placeholder="Search jobs, files, or messages"
                      onChange={(event) => setCleanupSearch(event.target.value)}
                    />
                  </label>
                  <label>
                    Sort Jobs
                    <select data-testid="cleanup-sort-select" value={cleanupSort} onChange={(event) => setCleanupSort(event.target.value as CleanupJobSort)}>
                      <option value="largest">Largest First</option>
                      <option value="newest">Newest First</option>
                      <option value="oldest">Oldest First</option>
                    </select>
                  </label>
                </div>
                <div className="job-progress-actions wrap-actions">
                  <button type="button" className="action-button secondary-button" data-testid="cleanup-bulk-scratch" onClick={() => void runBulkCleanup("scratch")} disabled={isBusy || hasActiveCleanupJobs || filteredCleanupJobs.length === 0}>
                    Clear Filtered Scratch
                  </button>
                  <button type="button" className="action-button secondary-button" data-testid="cleanup-bulk-output" onClick={() => void runBulkCleanup("output")} disabled={isBusy || hasActiveCleanupJobs || filteredCleanupJobs.length === 0}>
                    Delete Filtered Outputs
                  </button>
                  <button type="button" className="action-button secondary-button" data-testid="cleanup-bulk-all" onClick={() => void runBulkCleanup("all")} disabled={isBusy || hasActiveCleanupJobs || filteredCleanupJobs.length === 0}>
                    Clear Filtered Artifacts
                  </button>
                </div>
              <div className="cleanup-table-shell" data-testid="cleanup-jobs-table-shell">
                <table className="cleanup-jobs-table" data-testid="cleanup-jobs-table">
                  <thead>
                    <tr>
                      <th scope="col">State</th>
                      <th scope="col">Job Directory</th>
                      <th scope="col">Size</th>
                      <th scope="col">Last Update</th>
                      <th scope="col">Input File</th>
                      <th scope="col">Output File</th>
                      <th scope="col">Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredCleanupJobs.map((job) => {
                      const isExpanded = expandedCleanupJobIds.includes(job.id);
                      const combinedSizeBytes = cleanupJobTotalBytes(job);
                      return (
                        <Fragment key={job.id}>
                          <tr className="cleanup-jobs-row" data-testid={`cleanup-job-${job.id}`}>
                            <td>
                              <div className="cleanup-state-cell">
                                <span className="catalog-chip">{cleanupKindLabel(job.jobKind)}</span>
                                <strong>{job.state}</strong>
                                <span className="cleanup-row-message">{job.label}</span>
                              </div>
                            </td>
                            <td data-testid={`cleanup-directory-${job.id}`}>{jobDirectoryLabel(job.scratchPath)}</td>
                            <td data-testid={`cleanup-size-${job.id}`}>{formatBytes(combinedSizeBytes)}</td>
                            <td data-testid={`cleanup-updated-${job.id}`}>{formatRelativeTime(job.updatedAt)}</td>
                            <td data-testid={`cleanup-input-${job.id}`}>{pathLeaf(job.sourcePath)}</td>
                            <td data-testid={`cleanup-output-${job.id}`}>{pathLeaf(job.outputPath)}</td>
                            <td>
                              <button type="button" className="cleanup-expand-button" data-testid={`cleanup-expand-${job.id}`} onClick={() => toggleCleanupJobExpanded(job.id)}>
                                {isExpanded ? "Hide Details" : "Show Details"}
                              </button>
                            </td>
                          </tr>
                          {isExpanded ? (
                            <tr className="cleanup-jobs-detail-row" data-testid={`cleanup-details-row-${job.id}`}>
                              <td colSpan={7}>
                                <div className="cleanup-details-panel" data-testid={`cleanup-details-${job.id}`}>
                                  <div className="cleanup-details-grid">
                                    <span>Status Message: {job.message}</span>
                                    <span>Exact Updated: {formatExactTimestamp(job.updatedAt)}</span>
                                    <span>Phase: {job.phase}</span>
                                    <span>Recorded Frames: {job.recordedCount}</span>
                                    <span>Model: {job.modelId ?? "n/a"}</span>
                                    <span>Codec / Container: {job.codec ?? "n/a"} / {job.container ?? "n/a"}</span>
                                    <span>Scratch Size / Output Size: {formatBytes(job.scratchSizeBytes)} / {formatBytes(job.outputSizeBytes)}</span>
                                    <span>Average / Current Throughput: {formatFramesPerSecond(job.progress.averageFramesPerSecond)} / {formatFramesPerSecond(job.progress.rollingFramesPerSecond)}</span>
                                    <span>Elapsed / ETA: {formatElapsedSeconds(job.progress.elapsedSeconds)} / {formatElapsedSeconds(job.progress.estimatedRemainingSeconds)}</span>
                                    <span>Worker RAM / GPU Memory: {formatBytes(job.progress.processRssBytes ?? 0)} / {formatGpuMemory(job.progress.gpuMemoryUsedBytes, job.progress.gpuMemoryTotalBytes)}</span>
                                    <span>Stage Times: {formatStageTimings(job.progress)}</span>
                                    <span>Input Path: {job.sourcePath ?? "n/a"}</span>
                                    <span>Scratch Path: {job.scratchPath ?? "n/a"}</span>
                                    <span>Output Path: {job.outputPath ?? "n/a"}</span>
                                  </div>
                                  <div className="job-progress-actions wrap-actions">
                                    {job.outputPath ? (
                                      <button type="button" className="action-button secondary-button" data-testid={`cleanup-open-output-${job.id}`} onClick={() => void openMediaInDefaultApp(job.outputPath!)} disabled={isBusy}>
                                        Open Output
                                      </button>
                                    ) : null}
                                    {job.scratchPath ? (
                                      <button type="button" className="action-button secondary-button" data-testid={`cleanup-open-scratch-${job.id}`} onClick={() => void openMediaInDefaultApp(job.scratchPath!)} disabled={isBusy}>
                                        Open Scratch Folder
                                      </button>
                                    ) : null}
                                    {job.onStop ? (
                                      <button type="button" className="action-button secondary-button" data-testid={`cleanup-stop-${job.id}`} onClick={job.onStop} disabled={isBusy}>
                                        Stop Job
                                      </button>
                                    ) : null}
                                    {job.onClearScratch ? (
                                      <button type="button" className="action-button secondary-button" data-testid={`cleanup-clear-scratch-${job.id}`} onClick={job.onClearScratch} disabled={isBusy || isPipelineRunning}>
                                        Clear Job Scratch
                                      </button>
                                    ) : null}
                                    {job.onDeleteOutput ? (
                                      <button type="button" className="action-button secondary-button" data-testid={`cleanup-delete-output-${job.id}`} onClick={job.onDeleteOutput} disabled={isBusy || isPipelineRunning || isSourceConversionRunning}>
                                        Delete Job Output
                                      </button>
                                    ) : null}
                                  </div>
                                </div>
                              </td>
                            </tr>
                          ) : null}
                        </Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {filteredCleanupJobs.length === 0 ? (
                <p className="summary" data-testid="cleanup-empty-filter">No jobs match the current filter.</p>
              ) : null}
              </>
            ) : (
              <p className="summary">No tracked or historical managed jobs found yet.</p>
            )}
          </section>

          <section className="catalog-card cleanup-card" data-testid="cleanup-input-output-card">
            <div className="catalog-card-header">
              <strong>Current Input And Output</strong>
            </div>
            <div className="cleanup-job-grid">
              <article className="blind-sample-card">
                <div className="blind-sample-header">
                  <strong>Input</strong>
                </div>
                <div className="progress-stat-grid">
                  <span>Loaded: {source ? "Yes" : "No"}</span>
                  <span>Size: {formatBytes(sourcePathStats?.sizeBytes ?? 0)}</span>
                </div>
                <div className="job-progress-actions wrap-actions">
                  <button type="button" className="action-button secondary-button" data-testid="clear-input-button" onClick={clearLoadedInput} disabled={isBusy || isPipelineRunning || isBlindComparisonRunning || isSourceConversionRunning || !source}>
                    Clear Input
                  </button>
                  {source && isManagedArtifactPath(source.path) ? (
                    <button type="button" className="action-button secondary-button" data-testid="delete-input-file-button" onClick={() => void deleteManagedArtifact(source.path, "Input file", clearLoadedInput)} disabled={isBusy || isPipelineRunning || isBlindComparisonRunning || isSourceConversionRunning}>
                      Delete Input File
                    </button>
                  ) : null}
                </div>
              </article>
              <article className="blind-sample-card">
                <div className="blind-sample-header">
                  <strong>Output</strong>
                </div>
                <div className="progress-stat-grid">
                  <span>Ready: {result ? "Yes" : "No"}</span>
                  <span>Size: {formatBytes(outputPathStats?.sizeBytes ?? 0)}</span>
                </div>
                <div className="job-progress-actions wrap-actions">
                  <button type="button" className="action-button secondary-button" data-testid="clear-output-button" onClick={clearCurrentOutputSelection} disabled={isBusy || isPipelineRunning || !result}>
                    Clear Output
                  </button>
                  {result && isManagedArtifactPath(result.outputPath) ? (
                    <button type="button" className="action-button secondary-button" data-testid="delete-output-file-button" onClick={() => void deleteManagedArtifact(result.outputPath, "Output file", clearCurrentOutputSelection)} disabled={isBusy || isPipelineRunning}>
                      Delete Output File
                    </button>
                  ) : null}
                </div>
              </article>
            </div>
          </section>

          {scratchSummary ? (
            <section className="catalog-card cleanup-card" data-testid="scratch-summary-panel">
              <div className="catalog-card-header">
                <strong>Scratch Pools</strong>
              </div>
              <div className="progress-stat-grid">
                <span data-testid="jobs-pool-size">Jobs: {formatBytes(scratchSummary.jobsRoot.sizeBytes)}</span>
                <span data-testid="converted-pool-size">Converted Inputs: {formatBytes(scratchSummary.convertedSourcesRoot.sizeBytes)}</span>
                <span data-testid="previews-pool-size">Preview Proxies: {formatBytes(scratchSummary.sourcePreviewsRoot.sizeBytes)}</span>
              </div>
              <div className="job-progress-actions wrap-actions">
                <button type="button" className="action-button secondary-button" data-testid="clear-jobs-pool-button" onClick={() => void clearScratchPool(scratchSummary.jobsRoot.path, "Jobs scratch pool")} disabled={isBusy || isPipelineRunning || isSourceConversionRunning}>
                  Clear Jobs Pool
                </button>
                <button type="button" className="action-button secondary-button" data-testid="clear-converted-pool-button" onClick={() => void clearScratchPool(scratchSummary.convertedSourcesRoot.path, "Converted inputs pool")} disabled={isBusy || isPipelineRunning || isSourceConversionRunning}>
                  Clear Converted Pool
                </button>
                <button type="button" className="action-button secondary-button" data-testid="clear-previews-pool-button" onClick={() => void clearScratchPool(scratchSummary.sourcePreviewsRoot.path, "Preview proxy pool")} disabled={isBusy || isPipelineRunning || isSourceConversionRunning}>
                  Clear Preview Pool
                </button>
              </div>
            </section>
          ) : null}
        </ExpandablePanel>
      </section>

      {error ? <p data-testid="error-text" className="error-text">{error}</p> : null}
    </main>
  );
}
