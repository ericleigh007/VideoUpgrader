import { isTauri } from "@tauri-apps/api/core";
import { Fragment, useEffect, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { desktopApi } from "./lib/desktopApi";
import { getBackendDefinition, getBlindComparisonModels, getModelDefinition, getUiModels, getVisibleModels, type ModelDefinition } from "./lib/catalog";
import { defaultCropRect, planOutputFraming, resolveAspectRatio, resolveCropRect, type NormalizedCropRect } from "./lib/framing";
import { buildInterpolationWarning, interpolationTargetFpsOptions, isInterpolationEnabled } from "./lib/interpolation";
import {
  buildJobsWindowUrl,
  buildRepeatPipelineRequestEnvelope,
  JOBS_WINDOW_LABEL,
  parseRepeatPipelineRequestEnvelope,
  REPEAT_PIPELINE_REQUEST_STORAGE_KEY,
  resolveAppView,
} from "./lib/jobs";
import type { RepeatPipelineRequestAction, RepeatPipelineRequestEnvelope } from "./lib/jobs";
import type {
  AppConfig,
  AspectRatioPreset,
  InterpolationMode,
  InterpolationTargetFps,
  ManagedJobSummary,
  ManagedPipelineRunDetails,
  ModelId,
  OutputContainer,
  OutputMode,
  OutputSizingOptions,
  PathStats,
  PipelineProgress,
  PipelineJobStatus,
  PipelineMediaSummary,
  PipelineResourcePeaks,
  PipelineResult,
  PipelineStageTimings,
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
const blindComparisonDefaultCandidates = getBlindComparisonModels();
const blindComparisonAvailableModels = getVisibleModels().filter((model) => model.executionStatus === "runnable");
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

const APP_NAME = "VideoUpgrader";
const MOTION_SECTION_NAME = "Frame Rate Booster";
const OUTPUT_ROOT = "artifacts/video-upgrader/outputs";

function modelLaunchRequirement(model: ModelDefinition, runtime: RuntimeStatus | null): string | null {
  if (model.executionStatus !== "runnable") {
    return `${model.label} is cataloged but not implemented yet.`;
  }

  if (model.researchRuntime?.kind !== "external-command") {
    return null;
  }

  const runtimeStatus = runtime?.externalResearchRuntimes?.[model.value];
  if (runtimeStatus?.configured) {
    return null;
  }

  const commandEnvVar = runtimeStatus?.commandEnvVar || model.researchRuntime.commandEnvVar;
  return `${model.label} requires ${commandEnvVar} to be set before it can run.`;
}

function tauriWindowingAvailable(): boolean {
  return isTauri();
}

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

interface JobsWindowBounds {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface JobsWindowDragState {
  pointerOffsetX: number;
  pointerOffsetY: number;
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
  previewStartOffsetSeconds: number;
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
  testId?: string;
  children: ReactNode;
}

type TrackedJobEntry = {
  id: string;
  jobKind: string;
  label: string;
  state: "queued" | "running" | "paused" | "succeeded" | "failed" | "cancelled" | "interrupted";
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
  pipelineRunDetails: ManagedPipelineRunDetails | null;
  onPause: null | (() => void);
  onResume: null | (() => void);
  onStop: null | (() => void);
  onClearScratch: null | (() => void);
  onDeleteOutput: null | (() => void);
};

type CleanupJobFilter = "all" | "running" | "succeeded" | "cancelled" | "failed";
type CleanupSortColumn = "state" | "id" | "scratchSize" | "outputSize" | "updatedAt" | "input" | "output";
type CleanupSortDirection = "asc" | "desc";

type CleanupJobSort = {
  column: CleanupSortColumn;
  direction: CleanupSortDirection;
};

type ResolvedRepeatRequest = {
  request: RealesrganJobRequest;
  exact: boolean;
};

interface ProgressEventEntry {
  key: string;
  title: string;
  detail: string;
  percent: number;
  timestamp: number;
}

type PipelineLaunchState = "idle" | "starting" | PipelineJobStatus["state"];

const CLEANUP_FILTER_STORAGE_KEY = "videoupgrader.cleanup.filter";
const CLEANUP_SEARCH_STORAGE_KEY = "videoupgrader.cleanup.search";
const CLEANUP_SORT_STORAGE_KEY = "videoupgrader.cleanup.sort";
const RUN_SETTINGS_STORAGE_KEY = "videoupgrader.run.settings.v1";
const EMBEDDED_FULL_PREVIEW_CONTAINERS = new Set(["mp4"]);
const BLIND_COMPARISON_PREVIEW_CONTAINER: OutputContainer = "mp4";
const BLIND_COMPARISON_PREVIEW_CODEC: VideoCodec = "h264";
const AUTO_PREVIEW_UPGRADE_MAX_DURATION_SECONDS = 300;
const ACTIVE_PIPELINE_POLL_INTERVAL_MS = 250;
const MANAGED_JOBS_POLL_INTERVAL_MS = 5_000;
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
      <button
        type="button"
        className="expandable-panel-header"
        onClick={onToggle}
        aria-expanded={isOpen}
        data-testid={testId ? `${testId}-toggle` : undefined}
      >
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
  const extension = `.${container}`;
  return path.toLowerCase().endsWith(extension) ? path : `${path}${extension}`;
}

function defaultOutputPath(source: SourceVideoSummary | null, container: OutputContainer, modelId: string): string {
  const fileStem = source?.path.replace(/\\/g, "/").split("/").pop()?.replace(/\.[^.]+$/, "") ?? "video_upgrader_output";
  const sanitizedModelId = modelId.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  return `${OUTPUT_ROOT}/${fileStem}_${sanitizedModelId}.${container}`;
}

function pathLeaf(path: string | null | undefined): string {
  if (!path) {
    return "";
  }

  const normalized = path.replace(/\\/g, "/");
  const leaf = normalized.split("/").pop();
  return leaf && leaf.length > 0 ? leaf : normalized;
}

function pathLabel(path: string | null | undefined, emptyLabel: string): string {
  const leaf = pathLeaf(path);
  return leaf || emptyLabel;
}

const GENERIC_MANAGED_JOB_LABELS = new Set(["Upscale Export", "Upscale Job", "Source Conversion"]);

function managedJobLabel(label: string | null | undefined, sourcePath: string | null | undefined): string {
  const normalizedLabel = String(label ?? "").trim();
  if (normalizedLabel && !GENERIC_MANAGED_JOB_LABELS.has(normalizedLabel)) {
    return normalizedLabel;
  }

  const sourceLeaf = pathLeaf(sourcePath);
  if (sourceLeaf) {
    return sourceLeaf;
  }

  return normalizedLabel || "Unnamed Job";
}

function normalizeTimestampMillis(timestamp: number | string | null | undefined): number {
  const parsed = typeof timestamp === "number" ? timestamp : Number(timestamp);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 0;
  }
  return parsed < 1_000_000_000_000 ? parsed * 1000 : parsed;
}

function blindComparisonOutputPath(source: SourceVideoSummary, container: OutputContainer, modelId: string, anonymousLabel: string, runToken: string): string {
  const fileStem = source.path.replace(/\\/g, "/").split("/").pop()?.replace(/\.[^.]+$/, "") ?? "comparison_source";
  const sanitizedModelId = modelId.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  const sanitizedLabel = anonymousLabel.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  return `${OUTPUT_ROOT}/blind/${fileStem}_${runToken}_${sanitizedLabel}_${sanitizedModelId}.${container}`;
}

function parsePositiveIntegerInput(value: string): number | null {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function normalizePreviewStartOffsetSeconds(source: SourceVideoSummary | null, requestedOffsetSeconds: number | null | undefined): number {
  if (!source) {
    return 0;
  }

  const frameRate = Number.isFinite(source.frameRate) && source.frameRate > 0 ? source.frameRate : 0;
  const totalDuration = Number.isFinite(source.durationSeconds) && source.durationSeconds > 0 ? source.durationSeconds : 0;
  if (frameRate <= 0 || totalDuration <= 0) {
    return 0;
  }

  const clampedOffset = clamp(requestedOffsetSeconds ?? 0, 0, totalDuration);
  const totalFrameCount = Math.max(1, Math.round(totalDuration * frameRate));
  const startFrameIndex = Math.min(
    totalFrameCount - 1,
    Math.max(0, Math.ceil((clampedOffset * frameRate) - 0.000001)),
  );
  return startFrameIndex / frameRate;
}

function supportsEmbeddedFullLengthPreview(container: string | null | undefined): boolean {
  return EMBEDDED_FULL_PREVIEW_CONTAINERS.has(String(container ?? "").toLowerCase());
}

function previewMimeType(path: string | null | undefined): string | undefined {
  const normalized = String(path ?? "").toLowerCase();
  if (normalized.endsWith(".mp4")) {
    return "video/mp4";
  }
  if (normalized.endsWith(".webm")) {
    return "video/webm";
  }
  if (normalized.endsWith(".mov")) {
    return "video/quicktime";
  }
  if (normalized.endsWith(".mkv")) {
    return "video/x-matroska";
  }
  return undefined;
}

function clampUnit(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function clampCropRect(rect: NormalizedCropRect): NormalizedCropRect {
  const width = clampUnit(rect.width);
  const height = clampUnit(rect.height);
  return {
    width,
    height,
    left: Math.min(Math.max(0, rect.left), Math.max(0, 1 - width)),
    top: Math.min(Math.max(0, rect.top), Math.max(0, 1 - height)),
  };
}

function offsetCropRect(rect: NormalizedCropRect, deltaLeft: number, deltaTop: number): NormalizedCropRect {
  return clampCropRect({
    ...rect,
    left: rect.left + deltaLeft,
    top: rect.top + deltaTop,
  });
}

function resizeCropRect(rect: NormalizedCropRect, handle: Exclude<CropHandle, "move">, deltaX: number, deltaY: number, aspectRatio: number, sourceAspectRatio: number): NormalizedCropRect {
  const previewAspectRatio = sourceAspectRatio > 0 ? aspectRatio / sourceAspectRatio : aspectRatio;
  const anchorX = handle === "ne" || handle === "se" ? rect.left : rect.left + rect.width;
  const anchorY = handle === "sw" || handle === "se" ? rect.top : rect.top + rect.height;
  const signedDeltaX = (handle === "ne" || handle === "se" ? 1 : -1) * deltaX;
  const signedDeltaY = (handle === "sw" || handle === "se" ? 1 : -1) * deltaY;
  const width = Math.max(0.08, rect.width + signedDeltaX + signedDeltaY * previewAspectRatio);
  const nextWidth = Math.min(width, 1);
  const nextHeight = previewAspectRatio > 0 ? nextWidth / previewAspectRatio : nextWidth;
  const left = handle === "ne" || handle === "se" ? anchorX : anchorX - nextWidth;
  const top = handle === "sw" || handle === "se" ? anchorY : anchorY - nextHeight;
  return clampCropRect({ left, top, width: nextWidth, height: nextHeight });
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
      interpolatedFrames: 0,
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
      interpolatedFrames: 0,
      encodedFrames: 0,
      remuxedFrames: 0,
    },
    result: null,
    error: null,
  };
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
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
      interpolatedFrames: 0,
      encodedFrames: 0,
      remuxedFrames: 0,
    },
    result: null,
    error: null,
  };
}

function ratioFromCounts(value: number, total: number, forceComplete: boolean): number {
  if (forceComplete) {
    return 1;
  }
  if (total <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, value / total));
}

function formatDurationProgress(value: number): string {
  return `${(value / 1000).toFixed(1)}s`;
}

function formatElapsedSeconds(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || (value ?? 0) < 0) {
    return "calculating";
  }
  const rounded = Math.round(value ?? 0);
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

function formatClockTime(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || (value ?? 0) < 0) {
    return "0:00";
  }
  const rounded = Math.floor(value ?? 0);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const seconds = rounded % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function normalizeSourceCodec(value: string | null | undefined): VideoCodec | null {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (normalized.includes("hevc") || normalized.includes("h265") || normalized.includes("x265")) {
    return "h265";
  }
  if (normalized.includes("h264") || normalized.includes("avc") || normalized.includes("x264")) {
    return "h264";
  }
  return null;
}

function normalizeSourceContainer(value: string | null | undefined): OutputContainer | null {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized === "mp4" || normalized === "mkv" ? normalized : null;
}

function formatFramesPerSecond(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || (value ?? 0) <= 0) {
    return "calculating";
  }
  const resolved = value ?? 0;
  return `${resolved.toFixed(resolved >= 10 ? 1 : 2)} fps`;
}

function computePixelsPerSecond(media: PipelineMediaSummary | null | undefined, framesPerSecond: number | null | undefined): number | null {
  if (!media || !Number.isFinite(media.pixelCount) || media.pixelCount <= 0 || !Number.isFinite(framesPerSecond ?? NaN) || (framesPerSecond ?? 0) <= 0) {
    return null;
  }
  return media.pixelCount * (framesPerSecond ?? 0);
}

function formatPixelsPerSecond(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || (value ?? 0) <= 0) {
    return "calculating";
  }
  const resolved = value ?? 0;
  if (resolved >= 1_000_000) {
    return `${(resolved / 1_000_000).toFixed(resolved >= 10_000_000 ? 1 : 2)} MP/s (${Math.round(resolved).toLocaleString()} px/s)`;
  }
  return `${Math.round(resolved).toLocaleString()} px/s`;
}

function formatMediaSummary(media: PipelineMediaSummary | null | undefined): string {
  if (!media) {
    return "Unavailable";
  }
  return [
    `${media.width.toLocaleString()} x ${media.height.toLocaleString()}`,
    `${media.frameRate.toFixed(media.frameRate >= 10 ? 2 : 3)} fps`,
    `${media.frameCount.toLocaleString()} frames`,
    `${media.durationSeconds.toFixed(media.durationSeconds >= 10 ? 1 : 2)}s`,
  ].join(" • ");
}

function mediaFrameCount(media: PipelineMediaSummary | null | undefined): number | null {
  const frameCount = media?.frameCount;
  if (!Number.isFinite(frameCount ?? NaN) || (frameCount ?? 0) <= 0) {
    return null;
  }
  return Math.max(1, Math.round(frameCount ?? 0));
}

function derivePreviewFrameCount(durationSeconds: number | null | undefined, frameRate: number | null | undefined): number | null {
  if (!Number.isFinite(durationSeconds ?? NaN) || (durationSeconds ?? 0) <= 0 || !Number.isFinite(frameRate ?? NaN) || (frameRate ?? 0) <= 0) {
    return null;
  }
  return Math.max(1, Math.round((durationSeconds ?? 0) * (frameRate ?? 0)));
}

function formatStageTimingsSummary(timings: PipelineStageTimings | null | undefined): string | null {
  if (!timings) {
    return null;
  }

  const values = [
    { label: "extract", value: timings.extractSeconds },
    { label: "upscale", value: timings.upscaleSeconds },
    { label: "interpolate", value: timings.interpolateSeconds },
    { label: "encode", value: timings.encodeSeconds },
    { label: "remux", value: timings.remuxSeconds },
  ].filter((entry) => Number.isFinite(entry.value ?? NaN) && (entry.value ?? 0) >= 0);

  if (values.length === 0) {
    return null;
  }

  return values
    .filter((entry) => entry.label !== "interpolate" || (entry.value ?? 0) > 0)
    .map((entry) => `${entry.label} ${formatElapsedSeconds(entry.value)}`)
    .join(", ");
}

function formatStageTimings(progress: PipelineProgress): string {
  return formatStageTimingsSummary({
    extractSeconds: progress.extractStageSeconds ?? 0,
    upscaleSeconds: progress.upscaleStageSeconds ?? 0,
    interpolateSeconds: progress.interpolateStageSeconds ?? 0,
    encodeSeconds: progress.encodeStageSeconds ?? 0,
    remuxSeconds: progress.remuxStageSeconds ?? 0,
  }) ?? "unavailable";
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

function formatPeakRam(peaks: PipelineResourcePeaks | null | undefined): string {
  return formatBytes(peaks?.processRssBytes ?? 0);
}

function formatPeakGpuMemory(peaks: PipelineResourcePeaks | null | undefined): string {
  return formatGpuMemory(peaks?.gpuMemoryUsedBytes, peaks?.gpuMemoryTotalBytes);
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

function formatBitrateKbps(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || (value ?? 0) <= 0) {
    return "Unknown";
  }
  const resolved = value ?? 0;
  if (resolved >= 1000) {
    return `${(resolved / 1000).toFixed(resolved >= 10000 ? 1 : 2)} Mb/s`;
  }
  return `${resolved.toFixed(0)} kb/s`;
}

function formatSampleRate(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || (value ?? 0) <= 0) {
    return "Unknown";
  }
  return `${Math.round(value ?? 0).toLocaleString()} Hz`;
}

function greatestCommonDivisor(left: number, right: number): number {
  let a = Math.abs(Math.round(left));
  let b = Math.abs(Math.round(right));
  while (b !== 0) {
    const remainder = a % b;
    a = b;
    b = remainder;
  }
  return a || 1;
}

function formatAspectRatio(width: number, height: number): string {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return "Unknown";
  }
  const divisor = greatestCommonDivisor(width, height);
  return `${Math.round(width / divisor)}:${Math.round(height / divisor)}`;
}

function formatMediaLabel(value: string | null | undefined): string {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized) {
    return "Unknown";
  }

  const labels: Record<string, string> = {
    aac: "AAC",
    ac3: "AC-3",
    av1: "AV1",
    dts: "DTS",
    eac3: "E-AC-3",
    flac: "FLAC",
    h264: "H.264",
    h265: "H.265",
    hevc: "HEVC",
    mp3: "MP3",
    opus: "Opus",
    pcm_s16le: "PCM S16LE",
    pcm_s24le: "PCM S24LE",
    truehd: "TrueHD",
    vorbis: "Vorbis",
  };

  return labels[normalized] ?? normalized.replace(/[_-]+/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatTitleCase(value: string | null | undefined): string {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return "Unknown";
  }
  return normalized.replace(/\b\w/g, (char) => char.toUpperCase());
}

function buildSourceVideoSummary(source: SourceVideoSummary): string {
  return [
    formatMediaLabel(source.videoCodec),
    source.videoProfile?.trim() || null,
    `${source.width} x ${source.height}`,
    `${source.frameRate.toFixed(3)} fps`,
  ].filter(Boolean).join(" • ");
}

function buildSourceAudioSummary(source: SourceVideoSummary): string {
  if (!source.hasAudio) {
    return "No audio stream detected";
  }
  return [
    formatMediaLabel(source.audioCodec),
    source.audioProfile?.trim() || null,
    Number.isFinite(source.audioSampleRate ?? NaN) && (source.audioSampleRate ?? 0) > 0 ? formatSampleRate(source.audioSampleRate) : null,
    source.audioChannels ? formatTitleCase(source.audioChannels) : null,
  ].filter(Boolean).join(" • ");
}

function buildResultOutputSummary(result: PipelineResult, outputSizeBytes: number | null | undefined, workDirSizeBytes: number | null | undefined): string {
  return [
    `${result.frameCount.toLocaleString()} frames`,
    `${formatMediaLabel(result.codec)} / ${String(result.container ?? "").toUpperCase()}`,
    `Output ${formatBytes(outputSizeBytes ?? 0)}`,
    `Scratch ${formatBytes(workDirSizeBytes ?? 0)}`,
  ].join(" • ");
}

function formatQualityPresetLabel(value: string | null | undefined): string {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return "Unknown";
  }
  return qualityPresets.find((entry) => entry.value === normalized)?.label ?? formatTitleCase(normalized);
}

function formatPrecisionSourceLabel(value: string | null | undefined): string {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return "Unknown";
  }
  const labels: Record<string, string> = {
    "explicit-request": "Manual override",
    "preset-default": "Preset default",
    "backend-fixed": "Backend fixed",
  };
  return labels[normalized] ?? formatTitleCase(normalized);
}

function buildInterpolationDiagnosticsSummary(result: PipelineResult): string {
  if (!result.interpolationDiagnostics) {
    return "Interpolation details unavailable";
  }
  return [
    formatTitleCase(result.interpolationDiagnostics.mode),
    `${result.interpolationDiagnostics.sourceFps.toFixed(3)} -> ${result.interpolationDiagnostics.outputFps.toFixed(3)} fps`,
    `${result.interpolationDiagnostics.segmentCount} segments`,
    `${result.interpolationDiagnostics.segmentOverlapFrames} frame${result.interpolationDiagnostics.segmentOverlapFrames === 1 ? "" : "s"} overlap`,
  ].join(" • ");
}

function buildWorkerLogSummary(result: PipelineResult): string {
  return `${result.log.length.toLocaleString()} log line${result.log.length === 1 ? "" : "s"}`;
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
  if (progress.phase === "paused") {
    return "Pipeline paused";
  }
  if (progress.phase === "extracting") {
    return "Extracting source frames";
  }
  if (progress.phase === "upscaling") {
    return progress.batchCount ? "Upscaling the current batch" : "Upscaling extracted frames";
  }
  if (progress.phase === "interpolating") {
    return "Interpolating additional frames";
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

function formatPipelinePhaseLabel(phase: string): string {
  if (phase === "paused") {
    return "Paused";
  }
  if (phase === "extracting") {
    return "Extracting";
  }
  if (phase === "upscaling") {
    return "Upscaling";
  }
  if (phase === "interpolating") {
    return "Interpolating";
  }
  if (phase === "encoding") {
    return "Encoding";
  }
  if (phase === "remuxing") {
    return "Finalizing";
  }
  if (phase === "completed") {
    return "Completed";
  }
  if (phase === "failed") {
    return "Failed";
  }
  return "Preparing";
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
    progress.interpolatedFrames,
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

function isCleanupAttentionState(state: TrackedJobEntry["state"]): boolean {
  return state === "failed" || state === "interrupted";
}

function isActiveCleanupState(state: TrackedJobEntry["state"]): boolean {
  return state === "queued" || state === "running" || state === "paused";
}

function cleanupKindLabel(job: TrackedJobEntry): string {
  if (job.jobKind === "sourceConversion") {
    return "Conversion";
  }
  const hasInterpolation = (job.progress.interpolatedFrames ?? 0) > 0;
  const hasUpscale = (job.progress.upscaledFrames ?? 0) > 0 || Boolean(job.modelId);
  if (hasInterpolation && hasUpscale) {
    return "Upscale + Motion";
  }
  if (hasInterpolation) {
    return "Motion";
  }
  return "Upscale";
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

function safeLocalStorageRemove(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Ignore local storage failures.
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function defaultJobsWindowBounds(): JobsWindowBounds {
  if (typeof window === "undefined") {
    return { left: 32, top: 110, width: 720, height: 760 };
  }

  const width = Math.round(clamp(window.innerWidth * 0.42, 560, 820));
  const height = Math.round(clamp(window.innerHeight * 0.72, 420, 860));
  const left = Math.round(clamp(window.innerWidth - width - 24, 16, Math.max(16, window.innerWidth - width - 16)));
  const top = 96;
  return { left, top, width, height };
}

function clampJobsWindowBounds(bounds: JobsWindowBounds, width: number, height: number): JobsWindowBounds {
  if (typeof window === "undefined") {
    return bounds;
  }

  const clampedWidth = Math.round(clamp(width, 520, Math.max(520, window.innerWidth - 32)));
  const clampedHeight = Math.round(clamp(height, 360, Math.max(360, window.innerHeight - 32)));
  const maxLeft = Math.max(16, window.innerWidth - clampedWidth - 16);
  const maxTop = Math.max(16, window.innerHeight - clampedHeight - 16);
  return {
    left: Math.round(clamp(bounds.left, 16, maxLeft)),
    top: Math.round(clamp(bounds.top, 16, maxTop)),
    width: clampedWidth,
    height: clampedHeight,
  };
}

type PersistedRunSettings = {
  modelId: ModelId;
  outputMode: OutputMode;
  qualityPreset: QualityPreset;
  selectedGpuId: number | null;
  aspectRatioPreset: AspectRatioPreset;
  customAspectWidthInput: string;
  customAspectHeightInput: string;
  resolutionBasis: ResolutionBasis;
  targetWidthInput: string;
  targetHeightInput: string;
  codec: VideoCodec;
  container: OutputContainer;
  tileSize: number;
  crf: number;
  isUpscaleStepEnabled: boolean;
  isInterpolationStepEnabled: boolean;
  interpolationTargetFps: InterpolationTargetFps;
  pytorchRunner: PytorchRunner;
  previewMode: boolean;
  previewDurationInput: string;
  segmentDurationInput: string;
  isInputPanelOpen: boolean;
  isOutputPanelOpen: boolean;
  isBlindPanelOpen: boolean;
  isCleanupPanelOpen: boolean;
  jobsWindowLeft: number;
  jobsWindowTop: number;
  jobsWindowWidth: number;
  jobsWindowHeight: number;
};

function parsePersistedRunSettings(raw: string | null): Partial<PersistedRunSettings> {
  if (!raw) {
    return {};
  }

  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const settings: Partial<PersistedRunSettings> = {};

    if (typeof parsed.modelId === "string" && models.some((model) => model.value === parsed.modelId)) {
      settings.modelId = parsed.modelId;
    }
    if (parsed.outputMode === "preserveAspect4k" || parsed.outputMode === "cropTo4k" || parsed.outputMode === "native4x") {
      settings.outputMode = parsed.outputMode;
    }
    if (parsed.qualityPreset === "qualityMax" || parsed.qualityPreset === "qualityBalanced" || parsed.qualityPreset === "vramSafe") {
      settings.qualityPreset = parsed.qualityPreset;
    }
    if (typeof parsed.selectedGpuId === "number" || parsed.selectedGpuId === null) {
      settings.selectedGpuId = parsed.selectedGpuId;
    }
    if (parsed.aspectRatioPreset === "source" || parsed.aspectRatioPreset === "16:9" || parsed.aspectRatioPreset === "9:16" || parsed.aspectRatioPreset === "4:3" || parsed.aspectRatioPreset === "1:1" || parsed.aspectRatioPreset === "21:9" || parsed.aspectRatioPreset === "custom") {
      settings.aspectRatioPreset = parsed.aspectRatioPreset;
    }
    if (typeof parsed.customAspectWidthInput === "string") {
      settings.customAspectWidthInput = parsed.customAspectWidthInput;
    }
    if (typeof parsed.customAspectHeightInput === "string") {
      settings.customAspectHeightInput = parsed.customAspectHeightInput;
    }
    if (parsed.resolutionBasis === "exact" || parsed.resolutionBasis === "width" || parsed.resolutionBasis === "height") {
      settings.resolutionBasis = parsed.resolutionBasis;
    }
    if (typeof parsed.targetWidthInput === "string") {
      settings.targetWidthInput = parsed.targetWidthInput;
    }
    if (typeof parsed.targetHeightInput === "string") {
      settings.targetHeightInput = parsed.targetHeightInput;
    }
    if (parsed.codec === "h264" || parsed.codec === "h265") {
      settings.codec = parsed.codec;
    }
    if (parsed.container === "mp4" || parsed.container === "mkv") {
      settings.container = parsed.container;
    }
    if (typeof parsed.tileSize === "number" && Number.isFinite(parsed.tileSize) && parsed.tileSize >= 0) {
      settings.tileSize = parsed.tileSize;
    }
    if (typeof parsed.crf === "number" && Number.isFinite(parsed.crf)) {
      settings.crf = parsed.crf;
    }
    if (typeof parsed.isUpscaleStepEnabled === "boolean") {
      settings.isUpscaleStepEnabled = parsed.isUpscaleStepEnabled;
    }
    if (typeof parsed.isInterpolationStepEnabled === "boolean") {
      settings.isInterpolationStepEnabled = parsed.isInterpolationStepEnabled;
    }
    if (parsed.interpolationTargetFps === 30 || parsed.interpolationTargetFps === 60) {
      settings.interpolationTargetFps = parsed.interpolationTargetFps;
    }
    if (parsed.pytorchRunner === "torch" || parsed.pytorchRunner === "tensorrt") {
      settings.pytorchRunner = parsed.pytorchRunner;
    }
    if (typeof parsed.previewMode === "boolean") {
      settings.previewMode = parsed.previewMode;
    }
    if (typeof parsed.previewDurationInput === "string") {
      settings.previewDurationInput = parsed.previewDurationInput;
    }
    if (typeof parsed.segmentDurationInput === "string") {
      settings.segmentDurationInput = parsed.segmentDurationInput;
    }
    if (typeof parsed.isInputPanelOpen === "boolean") {
      settings.isInputPanelOpen = parsed.isInputPanelOpen;
    }
    if (typeof parsed.isOutputPanelOpen === "boolean") {
      settings.isOutputPanelOpen = parsed.isOutputPanelOpen;
    }
    if (typeof parsed.isBlindPanelOpen === "boolean") {
      settings.isBlindPanelOpen = parsed.isBlindPanelOpen;
    }
    if (typeof parsed.isCleanupPanelOpen === "boolean") {
      settings.isCleanupPanelOpen = parsed.isCleanupPanelOpen;
    }
    if (typeof parsed.jobsWindowLeft === "number" && Number.isFinite(parsed.jobsWindowLeft)) {
      settings.jobsWindowLeft = parsed.jobsWindowLeft;
    }
    if (typeof parsed.jobsWindowTop === "number" && Number.isFinite(parsed.jobsWindowTop)) {
      settings.jobsWindowTop = parsed.jobsWindowTop;
    }
    if (typeof parsed.jobsWindowWidth === "number" && Number.isFinite(parsed.jobsWindowWidth) && parsed.jobsWindowWidth > 0) {
      settings.jobsWindowWidth = parsed.jobsWindowWidth;
    }
    if (typeof parsed.jobsWindowHeight === "number" && Number.isFinite(parsed.jobsWindowHeight) && parsed.jobsWindowHeight > 0) {
      settings.jobsWindowHeight = parsed.jobsWindowHeight;
    }

    return settings;
  } catch {
    return {};
  }
}

function parseCleanupFilter(value: string | null): CleanupJobFilter {
  return value === "running" || value === "succeeded" || value === "cancelled" || value === "failed" ? value : "all";
}

function parseCleanupSort(value: string | null): CleanupJobSort {
  if (!value) {
    return { column: "scratchSize", direction: "desc" };
  }

  const [column, direction] = value.split(":", 2);
  const normalizedColumn = column === "size" ? "scratchSize" : column;
  if (
    (normalizedColumn === "state"
      || normalizedColumn === "id"
      || normalizedColumn === "scratchSize"
      || normalizedColumn === "outputSize"
      || normalizedColumn === "updatedAt"
      || normalizedColumn === "input"
      || normalizedColumn === "output")
    && (direction === "asc" || direction === "desc")
  ) {
    return { column: normalizedColumn, direction };
  }

  return { column: "scratchSize", direction: "desc" };
}

function toggleCleanupSort(currentSort: CleanupJobSort, column: CleanupSortColumn): CleanupJobSort {
  if (currentSort.column === column) {
    return {
      column,
      direction: currentSort.direction === "asc" ? "desc" : "asc",
    };
  }
  return {
    column,
    direction: "asc",
  };
}

function cleanupSortIndicator(currentSort: CleanupJobSort, column: CleanupSortColumn): string {
  if (currentSort.column !== column) {
    return "";
  }
  return currentSort.direction === "asc" ? "↑" : "↓";
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
  const directionFactor = sortMode.direction === "asc" ? 1 : -1;
  const compareText = (leftValue: string, rightValue: string): number => leftValue.localeCompare(rightValue, undefined, { sensitivity: "base" });

  let comparison = 0;
  switch (sortMode.column) {
    case "state":
      comparison = compareText(left.state, right.state) || compareText(left.label, right.label);
      break;
    case "id":
      comparison = compareText(left.id, right.id);
      break;
    case "scratchSize":
      comparison = left.scratchSizeBytes - right.scratchSizeBytes;
      break;
    case "outputSize":
      comparison = left.outputSizeBytes - right.outputSizeBytes;
      break;
    case "updatedAt":
      comparison = left.updatedAt - right.updatedAt;
      break;
    case "input":
      comparison = compareText(pathLabel(left.sourcePath, ""), pathLabel(right.sourcePath, ""));
      break;
    case "output":
      comparison = compareText(pathLabel(left.outputPath, ""), pathLabel(right.outputPath, ""));
      break;
  }

  if (comparison === 0) {
    comparison = right.updatedAt - left.updatedAt;
  }
  if (comparison === 0) {
    comparison = compareText(left.id, right.id);
  }
  return comparison * directionFactor;
}

function isManagedArtifactPath(path: string | null | undefined): boolean {
  if (!path) {
    return false;
  }

  return path.replace(/\\/g, "/").toLowerCase().includes("/artifacts/");
}

function buildDeleteConfirmation(title: string, path: string, details: string[]): string {
  return [title, "", `Path: ${path}`, ...details].join("\n");
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

function toggleIncludedModel(modelIds: ModelId[], targetModelId: ModelId): ModelId[] {
  return modelIds.includes(targetModelId)
    ? modelIds.filter((modelId) => modelId !== targetModelId)
    : [...modelIds, targetModelId];
}

export default function App() {
  const persistedRunSettings = parsePersistedRunSettings(safeLocalStorageGet(RUN_SETTINGS_STORAGE_KEY));
  const isJobsOnlyView = resolveAppView(window.location.search) === "jobs";
  const canOpenNativeJobsWindow = tauriWindowingAvailable() && !isJobsOnlyView;
  const [modelId, setModelId] = useState<ModelId>(persistedRunSettings.modelId ?? "realesrgan-x4plus");
  const [outputMode, setOutputMode] = useState<OutputMode>(persistedRunSettings.outputMode ?? "preserveAspect4k");
  const [qualityPreset, setQualityPreset] = useState<QualityPreset>(persistedRunSettings.qualityPreset ?? "qualityBalanced");
  const [selectedGpuId, setSelectedGpuId] = useState<number | null>(persistedRunSettings.selectedGpuId ?? null);
  const [aspectRatioPreset, setAspectRatioPreset] = useState<AspectRatioPreset>(persistedRunSettings.aspectRatioPreset ?? "16:9");
  const [customAspectWidthInput, setCustomAspectWidthInput] = useState<string>(persistedRunSettings.customAspectWidthInput ?? "16");
  const [customAspectHeightInput, setCustomAspectHeightInput] = useState<string>(persistedRunSettings.customAspectHeightInput ?? "9");
  const [resolutionBasis, setResolutionBasis] = useState<ResolutionBasis>(persistedRunSettings.resolutionBasis ?? "exact");
  const [targetWidthInput, setTargetWidthInput] = useState<string>(persistedRunSettings.targetWidthInput ?? "3840");
  const [targetHeightInput, setTargetHeightInput] = useState<string>(persistedRunSettings.targetHeightInput ?? "2160");
  const [cropRect, setCropRect] = useState<NormalizedCropRect | null>(null);
  const [codec, setCodec] = useState<VideoCodec>(persistedRunSettings.codec ?? "h264");
  const [container, setContainer] = useState<OutputContainer>(persistedRunSettings.container ?? "mp4");
  const [tileSize, setTileSize] = useState<number>(persistedRunSettings.tileSize ?? 0);
  const [crf, setCrf] = useState<number>(persistedRunSettings.crf ?? 18);
  const [isUpscaleStepEnabled, setIsUpscaleStepEnabled] = useState<boolean>(persistedRunSettings.isUpscaleStepEnabled ?? true);
  const [isInterpolationStepEnabled, setIsInterpolationStepEnabled] = useState<boolean>(persistedRunSettings.isInterpolationStepEnabled ?? false);
  const [interpolationTargetFps, setInterpolationTargetFps] = useState<InterpolationTargetFps>(persistedRunSettings.interpolationTargetFps ?? 60);
  const [pytorchRunner, setPytorchRunner] = useState<PytorchRunner>(persistedRunSettings.pytorchRunner ?? recommendedPytorchRunner(persistedRunSettings.modelId ?? "realesrgan-x4plus"));
  const [previewMode, setPreviewMode] = useState<boolean>(persistedRunSettings.previewMode ?? true);
  const [previewDurationInput, setPreviewDurationInput] = useState<string>(persistedRunSettings.previewDurationInput ?? "8");
  const [segmentDurationInput, setSegmentDurationInput] = useState<string>(persistedRunSettings.segmentDurationInput ?? "10");
  const [source, setSource] = useState<SourceVideoSummary | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [isPipelineLaunchPending, setIsPipelineLaunchPending] = useState(false);
  const [appConfig, setAppConfig] = useState<AppConfig | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activePipelineRequest, setActivePipelineRequest] = useState<RealesrganJobRequest | null>(null);
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
  const [isInputPanelOpen, setIsInputPanelOpen] = useState(persistedRunSettings.isInputPanelOpen ?? true);
  const [isOutputPanelOpen, setIsOutputPanelOpen] = useState(persistedRunSettings.isOutputPanelOpen ?? true);
  const [isBlindPanelOpen, setIsBlindPanelOpen] = useState(persistedRunSettings.isBlindPanelOpen ?? false);
  const [isCleanupPanelOpen, setIsCleanupPanelOpen] = useState(persistedRunSettings.isCleanupPanelOpen ?? false);
  const [jobsWindowBounds, setJobsWindowBounds] = useState<JobsWindowBounds>(() => {
    const defaults = defaultJobsWindowBounds();
    return clampJobsWindowBounds(
      {
        left: persistedRunSettings.jobsWindowLeft ?? defaults.left,
        top: persistedRunSettings.jobsWindowTop ?? defaults.top,
        width: persistedRunSettings.jobsWindowWidth ?? defaults.width,
        height: persistedRunSettings.jobsWindowHeight ?? defaults.height,
      },
      persistedRunSettings.jobsWindowWidth ?? defaults.width,
      persistedRunSettings.jobsWindowHeight ?? defaults.height,
    );
  });
  const [selectedBlindComparisonModelIds, setSelectedBlindComparisonModelIds] = useState<ModelId[]>(
    blindComparisonDefaultCandidates.map((candidate) => candidate.value),
  );
  const [blindComparisonStartOffsetSeconds, setBlindComparisonStartOffsetSeconds] = useState<number>(0);
  const [blindComparison, setBlindComparison] = useState<BlindComparisonState | null>(null);
  const [status, setStatus] = useState<string>("Idle");
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [isSavingRating, setIsSavingRating] = useState(false);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [jobsWindowDragState, setJobsWindowDragState] = useState<JobsWindowDragState | null>(null);
  const [isCropEditing, setIsCropEditing] = useState(false);
  const [comparisonZoom, setComparisonZoom] = useState<number>(3);
  const [comparisonFocusX, setComparisonFocusX] = useState<number>(50);
  const [comparisonFocusY, setComparisonFocusY] = useState<number>(50);
  const [comparisonFocusPresetId, setComparisonFocusPresetId] = useState<string>(comparisonFocusPresets[0]?.id ?? "dithering");
  const [comparisonCurrentTime, setComparisonCurrentTime] = useState<number>(0);
  const [comparisonDuration, setComparisonDuration] = useState<number>(0);
  const [comparisonPlaying, setComparisonPlaying] = useState<boolean>(false);
  const [isComparisonWorkspaceOpen, setIsComparisonWorkspaceOpen] = useState<boolean>(false);
  const [sourcePreviewPlaying, setSourcePreviewPlaying] = useState<boolean>(false);
  const [sourcePreviewCurrentTime, setSourcePreviewCurrentTime] = useState<number>(0);
  const [sourcePreviewDuration, setSourcePreviewDuration] = useState<number>(0);
  const [resolvedSourcePreviewUrl, setResolvedSourcePreviewUrl] = useState<string | null>(null);
  const [resolvedComparisonPreviewUrls, setResolvedComparisonPreviewUrls] = useState<Record<string, string>>({});
  const previewRef = useRef<HTMLDivElement | null>(null);
  const sourcePreviewVideoRef = useRef<HTMLVideoElement | null>(null);
  const comparisonSourceVideoRef = useRef<HTMLVideoElement | null>(null);
  const comparisonSampleVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const pipelineProgressSignatureRef = useRef<string | null>(null);
  const lastAppliedRepeatRequestAtRef = useRef<number>(0);
  const sourcePreviewAutoResumeRef = useRef(false);
  const resolvedSourcePreviewUrlRef = useRef<string | null>(null);
  const resolvedComparisonPreviewUrlsRef = useRef<Record<string, string>>({});
  const comparisonDesiredTimeRef = useRef<number>(0);
  const jobsPanelRef = useRef<HTMLDivElement | null>(null);

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
  const previewSrc = !isComparisonWorkspaceOpen && previewSourcePath ? (resolvedSourcePreviewUrl ?? desktopApi.toPreviewSrc(previewSourcePath)) : null;
  const sourcePreviewMimeType = previewMimeType(previewSourcePath);
  const resultPreviewSrc = result ? desktopApi.toPreviewSrc(result.outputPath) : null;
  const previewDurationSeconds = parsePositiveIntegerInput(previewDurationInput);
  const segmentDurationSeconds = parsePositiveIntegerInput(segmentDurationInput);
  const displayedWidth = resolutionBasis === "height" && framing ? String(framing.canvas.width) : targetWidthInput;
  const displayedHeight = resolutionBasis === "width" && framing ? String(framing.canvas.height) : targetHeightInput;
  const sourcePreviewSeekMax = Math.max(sourcePreviewDuration, 0.01);
  const normalizedBlindComparisonStartOffsetSeconds = normalizePreviewStartOffsetSeconds(source, blindComparisonStartOffsetSeconds);
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
  const isSourceConversionPaused = sourceConversionJob?.state === "paused";
  const isSourceConversionRunning = sourceConversionJob?.state === "queued" || sourceConversionJob?.state === "running" || sourceConversionJob?.state === "paused";
  const isBlockingSourceConversionRunning = Boolean(sourceConversionJob && isSourceConversionRunning && sourceConversionMode !== "preview");
  const activePrimaryJob = sourceConversionJob && isBlockingSourceConversionRunning
    ? sourceConversionJob
    : pipelineJob;
  const activeManagedJob = !activePrimaryJob
    ? [...managedJobs]
      .filter((job) => isActiveCleanupState(job.state as TrackedJobEntry["state"]))
      .sort((left, right) => normalizeTimestampMillis(right.updatedAt) - normalizeTimestampMillis(left.updatedAt))[0] ?? null
    : null;
  const activeDisplayProgress = activePrimaryJob?.progress ?? activeManagedJob?.progress ?? null;
  const progressPercent = activeDisplayProgress?.percent ?? (result ? 100 : 0);
  const progressMessage = activeDisplayProgress?.message ?? status;
  const isPipelinePaused = pipelineJob?.state === "paused";
  const isPipelineRunning = pipelineJob?.state === "queued" || pipelineJob?.state === "running" || pipelineJob?.state === "paused";
  const pipelineLaunchState: PipelineLaunchState = isPipelineLaunchPending
    ? "starting"
    : pipelineJob?.state ?? "idle";
  const pipelineLaunchStateLabel = pipelineLaunchState === "starting" ? "launching" : pipelineLaunchState;
  const hasRecoveredManagedJob = Boolean(activeManagedJob);
  const isBlindComparisonRunning = blindComparison?.state === "running";
  const selectedGpu = runtime?.availableGpus.find((gpu) => gpu.id === selectedGpuId) ?? null;
  const selectedModel = getModelDefinition(modelId);
  const selectedBackend = getBackendDefinition(selectedModel.backendId);
  const isSelectedModelImplemented = selectedModel.executionStatus === "runnable";
  const selectedModelLaunchRequirement = modelLaunchRequirement(selectedModel, runtime);
  const isSelectedModelLaunchable = selectedModelLaunchRequirement === null;
  const supportsPytorchRunner = selectedBackend.id === "pytorch-image-sr";
  const selectedModelRating = appConfig?.modelRatings[selectedModel.value]?.rating ?? null;
  const selectedBlindComparisonModels = blindComparisonAvailableModels.filter((model) => selectedBlindComparisonModelIds.includes(model.value));
  const comparisonEntries = blindComparison?.entries.filter((entry) => Boolean(entry.status.result?.outputPath)) ?? [];
  const comparisonFrameRateCandidates = [
    source?.frameRate ?? null,
    ...comparisonEntries.map((entry) => entry.status.result?.outputMedia?.frameRate ?? null),
  ].filter((value): value is number => Number.isFinite(value ?? NaN) && (value ?? 0) > 0);
  const comparisonFrameRate = comparisonFrameRateCandidates[0] ?? 0;
  const comparisonFrameCountCandidates = [
    source && blindComparison ? derivePreviewFrameCount(blindComparison.previewDurationSeconds, source.frameRate) : null,
    ...comparisonEntries.map((entry) => mediaFrameCount(entry.status.result?.outputMedia)),
  ].filter((value): value is number => Number.isFinite(value ?? NaN) && (value ?? 0) > 0);
  const comparisonFrameCount = comparisonFrameCountCandidates.length > 0 ? Math.min(...comparisonFrameCountCandidates) : 0;
  const comparisonTimelineMax = comparisonFrameCount > 0 ? comparisonFrameCount - 1 : 0;
  const comparisonCurrentFrame = comparisonFrameRate > 0
    ? Math.min(comparisonTimelineMax, Math.max(0, Math.round(comparisonCurrentTime * comparisonFrameRate)))
    : 0;
  const comparisonSourcePreviewPath = source ? (source.previewPath || source.path) : null;
  const comparisonSourcePreviewSrc = comparisonSourcePreviewPath ? (resolvedSourcePreviewUrl ?? desktopApi.toPreviewSrc(comparisonSourcePreviewPath)) : "";
  const comparisonSourcePreviewMimeType = previewMimeType(comparisonSourcePreviewPath);
  const comparisonPreviewLoadSignature = [
    comparisonSourcePreviewSrc,
    ...comparisonEntries.map((entry) => resolvedComparisonPreviewUrls[entry.sampleId] ?? (entry.status.result?.outputPath ?? "")),
  ].join("|");
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
  const interpolationMode: InterpolationMode = !isUpscaleStepEnabled && isInterpolationStepEnabled
    ? "interpolateOnly"
    : isUpscaleStepEnabled && isInterpolationStepEnabled
      ? "afterUpscale"
      : "off";
  const hasEnabledPipelineStep = isUpscaleStepEnabled || isInterpolationStepEnabled;
  const interpolationEnabled = isInterpolationEnabled(interpolationMode);
  const selectedQualityPresetLabel = qualityPresets.find((entry) => entry.value === qualityPreset)?.label ?? qualityPreset;
  const selectedCodecLabel = codecs.find((entry) => entry.value === codec)?.label ?? codec.toUpperCase();
  const selectedContainerLabel = containers.find((entry) => entry.value === container)?.label ?? container.toUpperCase();
  const encodingDetailsSummary = `${selectedCodecLabel} / ${selectedContainerLabel} • CRF ${crf} • ${selectedQualityPresetLabel} • Tile ${tileSize > 0 ? tileSize : "Auto"}`;
  const compactPipelineLabel = [
    isUpscaleStepEnabled ? selectedModel.label : null,
    interpolationEnabled ? `Interpolation ${interpolationTargetFps} fps` : null,
    hasEnabledPipelineStep ? selectedCodecLabel : null,
  ].filter((entry): entry is string => Boolean(entry)).join(" -> ");
  const matchedInputCodec = normalizeSourceCodec(source?.videoCodec);
  const matchedInputContainer = normalizeSourceContainer(source?.container);
  const canMatchInputFormat = Boolean(matchedInputCodec || matchedInputContainer);
  const matchInputFormatSummary = source
    ? canMatchInputFormat
      ? `Input video detected as ${source.videoCodec.toUpperCase()} in ${source.container.toUpperCase()}. Match Input will apply supported export settings.`
      : `Input video detected as ${source.videoCodec.toUpperCase()} in ${source.container.toUpperCase()}. Match Input is unavailable because that format is not supported for export.`
    : "Load a source file to match its codec and container where supported.";
  const isRunDisabled = isBusy
    || !source
    || isBlindComparisonRunning
    || isPipelineRunning
    || isBlockingSourceConversionRunning
    || !hasEnabledPipelineStep
    || (isUpscaleStepEnabled && !isSelectedModelLaunchable);
  const isBlindComparisonDisabled = isBusy || !source || isBlindComparisonRunning || isPipelineRunning || isBlockingSourceConversionRunning || selectedBlindComparisonModelIds.length < 2;
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
      pipelineRunDetails: activePipelineRequest ? { request: activePipelineRequest } : null,
      onPause: sourceConversionJob.state === "queued" || sourceConversionJob.state === "running" ? () => {
        void pauseSourceConversion();
      } : null,
      onResume: sourceConversionJob.state === "paused" ? () => {
        void resumeSourceConversion();
      } : null,
      onStop: isSourceConversionRunning ? () => {
        void cancelSourceConversion();
      } : null,
      onClearScratch: null as null | (() => void),
      onDeleteOutput: sourceConversionJob.result?.path && isManagedArtifactPath(sourceConversionJob.result.path)
        ? () => void deleteManagedArtifact(sourceConversionJob.result!.path, "Converted input", clearLoadedInput, [
          "Removes the generated MP4 copy created for conversion or preview compatibility.",
          "Clears it from the current input slot if this converted file is loaded.",
        ])
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
      pipelineRunDetails: null,
      onPause: pipelineJob.state === "queued" || pipelineJob.state === "running" ? () => {
        void pausePipeline();
      } : null,
      onResume: pipelineJob.state === "paused" ? () => {
        void resumePipeline();
      } : null,
      onStop: isPipelineRunning ? () => {
        void cancelPipeline();
      } : null,
      onClearScratch: result?.workDir && isManagedArtifactPath(result.workDir)
        ? () => void deleteManagedArtifact(result.workDir, "Job scratch", () => setWorkDirStats(null), [
          "Deletes intermediate extracted, upscaled, interpolated, and staging files for this job.",
          "The job history entry remains, but reruns will need to regenerate these artifacts.",
        ])
        : null,
      onDeleteOutput: result?.outputPath && isManagedArtifactPath(result.outputPath)
        ? () => void deleteManagedArtifact(result.outputPath, "Job output", clearCurrentOutputSelection, [
          "Deletes the exported video file for this job.",
          "Clears the current output selection if it points at this file.",
        ])
        : null,
    } : null,
  ];
  const trackedJobs = trackedJobCandidates.filter((entry): entry is TrackedJobEntry => entry !== null);
  const historicalJobs = managedJobs.map<TrackedJobEntry>((job) => ({
    id: job.jobId,
    jobKind: job.jobKind,
    label: managedJobLabel(job.label, job.sourcePath),
    state: job.state as TrackedJobEntry["state"],
    phase: job.progress.phase,
    progress: job.progress,
    modelId: job.modelId,
    codec: job.codec,
    container: job.container,
    recordedCount: job.recordedCount,
    message: job.progress.message,
    updatedAt: normalizeTimestampMillis(job.updatedAt),
    sourcePath: job.sourcePath ?? job.pipelineRunDetails?.request.sourcePath ?? null,
    scratchPath: job.scratchPath,
    scratchSizeBytes: job.progress.scratchSizeBytes ?? job.scratchStats?.sizeBytes ?? 0,
    outputPath: job.outputPath ?? job.pipelineRunDetails?.request.outputPath ?? null,
    outputSizeBytes: job.progress.outputSizeBytes ?? job.outputStats?.sizeBytes ?? 0,
    pipelineRunDetails: job.pipelineRunDetails ?? null,
    onPause: null,
    onResume: null,
    onStop: null,
    onClearScratch: job.scratchPath && isManagedArtifactPath(job.scratchPath)
      ? () => void deleteManagedArtifact(job.scratchPath!, `${job.label} scratch`, () => {}, [
        "Deletes this job's stored intermediate artifacts and scratch directory.",
        "Historical metadata remains so the run can still be inspected later.",
      ])
      : null,
    onDeleteOutput: job.outputPath && isManagedArtifactPath(job.outputPath)
      ? () => void deleteManagedArtifact(job.outputPath!, `${job.label} output`, () => {}, job.jobKind === "sourceConversion" ? [
        "Removes the converted source file created by the app.",
        "Use this when you no longer need the compatibility copy in artifacts/runtime/converted-sources.",
      ] : [
        "Deletes the exported output file for this historical job.",
        "The job record remains, but the output file itself will be gone.",
      ])
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
      pipelineRunDetails: job.pipelineRunDetails ?? existing?.pipelineRunDetails ?? null,
      onPause: job.onPause ?? existing?.onPause ?? null,
      onResume: job.onResume ?? existing?.onResume ?? null,
      onStop: job.onStop ?? existing?.onStop ?? null,
      onClearScratch: job.onClearScratch ?? existing?.onClearScratch ?? null,
      onDeleteOutput: job.onDeleteOutput ?? existing?.onDeleteOutput ?? null,
    });
  }
  const cleanupJobs = Array.from(cleanupJobsById.values()).sort((left, right) => sortCleanupJobs(left, right, cleanupSort));
  const cleanupStateCounts = {
    all: cleanupJobs.length,
    running: cleanupJobs.filter((job) => isActiveCleanupState(job.state)).length,
    succeeded: cleanupJobs.filter((job) => job.state === "succeeded").length,
    cancelled: cleanupJobs.filter((job) => job.state === "cancelled").length,
    failed: cleanupJobs.filter((job) => isCleanupAttentionState(job.state)).length,
  };
  const filteredCleanupJobs = cleanupJobs.filter((job) => {
    if (cleanupFilter === "all") {
      return matchesCleanupSearch(job, cleanupSearch);
    }
    if (cleanupFilter === "running") {
      return isActiveCleanupState(job.state) && matchesCleanupSearch(job, cleanupSearch);
    }
    if (cleanupFilter === "failed") {
      return isCleanupAttentionState(job.state) && matchesCleanupSearch(job, cleanupSearch);
    }
    return job.state === cleanupFilter && matchesCleanupSearch(job, cleanupSearch);
  });
  const hasActiveCleanupJobs = cleanupJobs.some((job) => isActiveCleanupState(job.state));
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
      id: "interpolate",
      label: "Interpolate",
      value: ratioFromCounts(pipelineJob.progress.interpolatedFrames, pipelineJob.progress.totalFrames, pipelineJob.state === "succeeded"),
      summary: `${pipelineJob.progress.interpolatedFrames}/${pipelineJob.progress.totalFrames || "?"}`,
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
  const pipelinePhaseLabel = pipelineJob ? formatPipelinePhaseLabel(pipelineJob.progress.phase) : null;
  const pipelineLastUpdateLabel = lastPipelineProgressAt ? formatRelativeTime(lastPipelineProgressAt) : "waiting for first update";
  const compactStatusTitle = isPipelineLaunchPending
    ? "Pipeline Starting"
    : isPipelinePaused
    ? "Pipeline Paused"
    : isPipelineRunning
      ? "Pipeline Running"
      : isSourceConversionPaused
        ? "Source Conversion Paused"
        : isSourceConversionRunning
          ? "Source Conversion Running"
          : activeManagedJob?.state === "paused"
            ? "Recovered Paused Job"
            : hasRecoveredManagedJob
              ? "Recovered Running Job"
      : result
        ? "Last Output Ready"
        : "Ready To Configure";
  const compactPhaseBars = pipelinePhaseBars.filter((entry) => entry.id === "upscale" || entry.id === "interpolate");
  const activePipelineVisualStep = (() => {
    if (!pipelineJob) {
      return null;
    }

    switch (pipelineJob.progress.phase) {
      case "interpolating":
        return "interpolate";
      case "encoding":
      case "remuxing":
      case "completed":
        return pipelineJob.progress.interpolatedFrames > 0 ? "interpolate" : "upscale";
      default:
        return "upscale";
    }
  })();
  const compactStatusDetail = isPipelineLaunchPending && !pipelineJob
    ? "Launch accepted • waiting for worker job id"
    : pipelineJob
    ? `${pipelinePhaseLabel ?? "Preparing"} • ${pipelineJob.progress.percent}% • ${activePrimaryJob && (activePrimaryJob.progress.estimatedRemainingSeconds ?? 0) > 0 ? `ETA ${formatElapsedSeconds(activePrimaryJob.progress.estimatedRemainingSeconds)}` : "ETA pending"}`
    : isSourceConversionPaused
      ? "Source conversion paused"
      : isSourceConversionRunning
        ? "Preparing source preview"
        : activeManagedJob
          ? `${formatPipelinePhaseLabel(activeManagedJob.progress.phase)} • ${activeManagedJob.progress.percent}% • ${(activeManagedJob.progress.estimatedRemainingSeconds ?? 0) > 0 ? `ETA ${formatElapsedSeconds(activeManagedJob.progress.estimatedRemainingSeconds)}` : "ETA pending"}`
      : result
        ? `Ready • ${pathLeaf(result.outputPath)}`
        : "Ready to configure";
  const topStatusPauseAction = isPipelinePaused
    ? () => {
        void resumePipeline();
      }
    : isPipelineRunning
      ? () => {
          void pausePipeline();
        }
      : isSourceConversionPaused
        ? () => {
            void resumeSourceConversion();
          }
        : isSourceConversionRunning
          ? () => {
              void pauseSourceConversion();
            }
          : null;
  const topStatusStopAction = isPipelineRunning
    ? () => {
        void cancelPipeline();
      }
    : isSourceConversionRunning
      ? () => {
          void cancelSourceConversion();
        }
      : null;
  const runtimeFactsSummary = runtime
    ? `${runtime.availableGpus.length > 0 ? `${runtime.availableGpus.length} GPU${runtime.availableGpus.length === 1 ? "" : "s"}` : "No GPUs detected"} • ${selectedGpu ? `GPU ${selectedGpu.id}` : "Auto GPU"}`
    : "Runtime assets download on first use";

  function toggleCleanupPanel(): void {
    setIsCleanupPanelOpen((current) => !current);
  }

  async function openJobsWindow(): Promise<void> {
    if (!canOpenNativeJobsWindow) {
      toggleCleanupPanel();
      return;
    }

    try {
      const { WebviewWindow } = await import("@tauri-apps/api/webviewWindow");
      const existingWindow = await WebviewWindow.getByLabel(JOBS_WINDOW_LABEL);
      if (existingWindow) {
        await existingWindow.show();
        await existingWindow.setFocus();
        setStatus("Jobs window ready.");
        return;
      }

      const jobsUrl = buildJobsWindowUrl(window.location);

      const jobsWindow = new WebviewWindow(JOBS_WINDOW_LABEL, {
        url: jobsUrl,
        title: `${APP_NAME} Jobs`,
        width: 1180,
        height: 900,
        minWidth: 840,
        minHeight: 620,
        resizable: true,
        focus: true,
      });

      await new Promise<void>((resolve, reject) => {
        let settled = false;
        void jobsWindow.once("tauri://created", () => {
          if (!settled) {
            settled = true;
            resolve();
          }
        });
        void jobsWindow.once("tauri://error", (event) => {
          if (!settled) {
            settled = true;
            reject(new Error(String(event.payload ?? "Failed to open the jobs window.")));
          }
        });
      });

      setStatus("Jobs window opened.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to open the jobs window.");
    }
  }

  async function closeJobsWindow(): Promise<void> {
    if (!isJobsOnlyView) {
      toggleCleanupPanel();
      return;
    }

    try {
      const { getCurrentWebviewWindow } = await import("@tauri-apps/api/webviewWindow");
      await getCurrentWebviewWindow().close();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to close the jobs window.");
    }
  }

  function beginJobsWindowDrag(event: ReactMouseEvent<HTMLElement>): void {
    if (isJobsOnlyView) {
      return;
    }

    const target = event.target as HTMLElement;
    if (target.closest("button, input, select, textarea, summary, a")) {
      return;
    }

    const panel = jobsPanelRef.current;
    if (!panel) {
      return;
    }

    const bounds = panel.getBoundingClientRect();
    setJobsWindowDragState({
      pointerOffsetX: event.clientX - bounds.left,
      pointerOffsetY: event.clientY - bounds.top,
    });
  }

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
      comparisonSampleVideoRefs.current = {};
      if (isComparisonWorkspaceOpen) {
        setIsComparisonWorkspaceOpen(false);
      }
      return;
    }

    const activeSampleIds = new Set(comparisonEntries.map((entry) => entry.sampleId));
    comparisonSampleVideoRefs.current = Object.fromEntries(
      Object.entries(comparisonSampleVideoRefs.current).filter(([sampleId]) => activeSampleIds.has(sampleId)),
    );
  }, [comparisonEntries, isComparisonWorkspaceOpen]);

  useEffect(() => {
    if (!isComparisonWorkspaceOpen && comparisonPlaying) {
      setComparisonPlaying(false);
    }
  }, [comparisonPlaying, isComparisonWorkspaceOpen]);

  useEffect(() => {
    if (!isComparisonWorkspaceOpen) {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      if (comparisonFrameRate > 0 && comparisonCurrentFrame > 0) {
        syncComparisonFrame(comparisonCurrentFrame);
        return;
      }

      if (comparisonCurrentTime > 0) {
        syncComparisonTime(comparisonCurrentTime);
      }
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [comparisonCurrentFrame, comparisonCurrentTime, comparisonFrameRate, comparisonPreviewLoadSignature, isComparisonWorkspaceOpen]);

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
    setBlindComparisonStartOffsetSeconds(0);
  }, [source?.path]);

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

        if (nextJob.state === "paused") {
          setStatus(nextJob.progress.message || "Pipeline paused.");
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
    }, ACTIVE_PIPELINE_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [activeJobId]);

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

  useEffect(() => () => {
    Object.values(resolvedComparisonPreviewUrlsRef.current).forEach((url) => {
      if (url.startsWith("blob:")) {
        URL.revokeObjectURL(url);
      }
    });
  }, []);

  function replaceResolvedSourcePreviewUrl(nextUrl: string | null): void {
    const currentUrl = resolvedSourcePreviewUrlRef.current;
    if (currentUrl?.startsWith("blob:") && currentUrl !== nextUrl) {
      URL.revokeObjectURL(currentUrl);
    }
    resolvedSourcePreviewUrlRef.current = nextUrl;
    setResolvedSourcePreviewUrl(nextUrl);
  }

  function replaceResolvedComparisonPreviewUrls(nextUrls: Record<string, string>): void {
    for (const [sampleId, currentUrl] of Object.entries(resolvedComparisonPreviewUrlsRef.current)) {
      if (currentUrl.startsWith("blob:") && nextUrls[sampleId] !== currentUrl) {
        URL.revokeObjectURL(currentUrl);
      }
    }

    resolvedComparisonPreviewUrlsRef.current = nextUrls;
    setResolvedComparisonPreviewUrls(nextUrls);
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
    let cancelled = false;

    async function resolveComparisonPreviewUrls(): Promise<void> {
      if (comparisonEntries.length === 0) {
        replaceResolvedComparisonPreviewUrls({});
        return;
      }

      const nextPairs = await Promise.all(comparisonEntries.map(async (entry) => {
        const outputPath = entry.status.result?.outputPath;
        if (!outputPath) {
          return [entry.sampleId, null] as const;
        }

        try {
          const nextUrl = await desktopApi.loadPreviewUrl(outputPath);
          return [entry.sampleId, nextUrl] as const;
        } catch {
          return [entry.sampleId, null] as const;
        }
      }));

      const nextUrls = Object.fromEntries(nextPairs.filter((entry): entry is readonly [string, string] => typeof entry[1] === "string"));
      if (cancelled) {
        Object.values(nextUrls).forEach((url) => {
          if (url.startsWith("blob:")) {
            URL.revokeObjectURL(url);
          }
        });
        return;
      }

      replaceResolvedComparisonPreviewUrls(nextUrls);
    }

    void resolveComparisonPreviewUrls();

    return () => {
      cancelled = true;
    };
  }, [comparisonEntries]);

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
            setActivePipelineRequest(null);
            setBlindComparison(null);
            setIsCropEditing(false);
            setIsComparisonWorkspaceOpen(false);
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

        if (nextJob.state === "paused") {
          setStatus(nextJob.progress.message || "Source conversion paused.");
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

  async function loadManagedArtifacts(includeScratchSummary = true): Promise<{
    scratchSummary: ScratchStorageSummary | null;
    managedJobs: ManagedJobSummary[];
  }> {
    const [nextScratch, nextManagedJobs] = await Promise.all([
      includeScratchSummary ? desktopApi.getScratchStorageSummary() : Promise.resolve(null),
      desktopApi.listManagedJobs(),
    ]);

    return {
      scratchSummary: nextScratch,
      managedJobs: nextManagedJobs,
    };
  }

  async function loadSelectedPathStats(): Promise<{
    sourcePathStats: PathStats | null;
    outputPathStats: PathStats | null;
    workDirStats: PathStats | null;
  }> {
    const [nextSource, nextOutput, nextWorkDir] = await Promise.all([
      source ? desktopApi.getPathStats(source.path) : Promise.resolve(null),
      result ? desktopApi.getPathStats(result.outputPath) : Promise.resolve(outputPath ? desktopApi.getPathStats(outputPath) : null),
      result ? desktopApi.getPathStats(result.workDir) : Promise.resolve(null),
    ]);

    return {
      sourcePathStats: nextSource,
      outputPathStats: nextOutput,
      workDirStats: nextWorkDir,
    };
  }

  function applyManagedArtifacts(snapshot: {
    scratchSummary: ScratchStorageSummary | null;
    managedJobs: ManagedJobSummary[];
  }): void {
    if (snapshot.scratchSummary) {
      setScratchSummary(snapshot.scratchSummary);
    }
    setManagedJobs(snapshot.managedJobs);
  }

  function applySelectedPathStats(snapshot: {
    sourcePathStats: PathStats | null;
    outputPathStats: PathStats | null;
    workDirStats: PathStats | null;
  }): void {
    setSourcePathStats(snapshot.sourcePathStats);
    setOutputPathStats(snapshot.outputPathStats);
    setWorkDirStats(snapshot.workDirStats);
  }

  async function refreshManagedWorkspaceState(options?: { includeScratchSummary?: boolean }): Promise<void> {
    const includeScratchSummary = options?.includeScratchSummary ?? true;
    const [managedSnapshot, pathStatsSnapshot] = await Promise.all([
      loadManagedArtifacts(includeScratchSummary),
      loadSelectedPathStats(),
    ]);
    applyManagedArtifacts(managedSnapshot);
    applySelectedPathStats(pathStatsSnapshot);
  }

  useEffect(() => {
    let disposed = false;

    async function refreshPathStats(): Promise<void> {
      try {
        const snapshot = await loadSelectedPathStats();

        if (disposed) {
          return;
        }

        applySelectedPathStats(snapshot);
      } catch {
        if (!disposed) {
          setSourcePathStats(null);
          setOutputPathStats(null);
          setWorkDirStats(null);
        }
      }
    }

    void refreshPathStats();
    return () => {
      disposed = true;
    };
  }, [outputPath, result, source]);

  useEffect(() => {
    let disposed = false;
    const includeScratchSummary = isJobsOnlyView || isCleanupPanelOpen;

    async function refreshStorage(): Promise<void> {
      try {
        const snapshot = await loadManagedArtifacts(includeScratchSummary);

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
    const shouldPollStorage = isJobsOnlyView || isCleanupPanelOpen || isPipelineRunning || isSourceConversionRunning;
    const intervalId = shouldPollStorage
      ? window.setInterval(() => {
        void refreshStorage();
      }, MANAGED_JOBS_POLL_INTERVAL_MS)
      : null;

    return () => {
      disposed = true;
      if (intervalId !== null) {
        window.clearInterval(intervalId);
      }
    };
  }, [isCleanupPanelOpen, isJobsOnlyView, isPipelineRunning, isSourceConversionRunning]);

  useEffect(() => {
    safeLocalStorageSet(CLEANUP_FILTER_STORAGE_KEY, cleanupFilter);
  }, [cleanupFilter]);

  useEffect(() => {
    safeLocalStorageSet(CLEANUP_SEARCH_STORAGE_KEY, cleanupSearch);
  }, [cleanupSearch]);

  useEffect(() => {
    safeLocalStorageSet(CLEANUP_SORT_STORAGE_KEY, `${cleanupSort.column}:${cleanupSort.direction}`);
  }, [cleanupSort]);

  useEffect(() => {
    if (isJobsOnlyView) {
      return undefined;
    }

    const applyReplayEnvelope = (raw: string | null) => {
      if (!raw) {
        return;
      }

      try {
        const envelope = parseRepeatPipelineRequestEnvelope(raw);
        if (!envelope?.request || envelope.requestedAt <= lastAppliedRepeatRequestAtRef.current) {
          return;
        }

        lastAppliedRepeatRequestAtRef.current = envelope.requestedAt;
        void (async () => {
          try {
            await applyRepeatedPipelineRequest(envelope.request);
            if (envelope.action === "restart") {
              setStatus(`Restarting ${pathLeaf(envelope.request.sourcePath)}...`);
              await startPipelineFromRequest(envelope.request, { ensureRuntime: false, queuedStatus: "Restarted job queued." });
            }
          } finally {
            safeLocalStorageRemove(REPEAT_PIPELINE_REQUEST_STORAGE_KEY);
          }
        })();
      } catch {
        safeLocalStorageRemove(REPEAT_PIPELINE_REQUEST_STORAGE_KEY);
      }
    };

    applyReplayEnvelope(safeLocalStorageGet(REPEAT_PIPELINE_REQUEST_STORAGE_KEY));

    const handleStorage = (event: StorageEvent) => {
      if (event.key !== REPEAT_PIPELINE_REQUEST_STORAGE_KEY) {
        return;
      }
      applyReplayEnvelope(event.newValue);
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, [isJobsOnlyView]);

  useEffect(() => {
    const nextSettings: PersistedRunSettings = {
      modelId,
      outputMode,
      qualityPreset,
      selectedGpuId,
      aspectRatioPreset,
      customAspectWidthInput,
      customAspectHeightInput,
      resolutionBasis,
      targetWidthInput,
      targetHeightInput,
      codec,
      container,
      tileSize,
      crf,
      isUpscaleStepEnabled,
      isInterpolationStepEnabled,
      interpolationTargetFps,
      pytorchRunner,
      previewMode,
      previewDurationInput,
      segmentDurationInput,
      isInputPanelOpen,
      isOutputPanelOpen,
      isBlindPanelOpen,
      isCleanupPanelOpen,
      jobsWindowLeft: jobsWindowBounds.left,
      jobsWindowTop: jobsWindowBounds.top,
      jobsWindowWidth: jobsWindowBounds.width,
      jobsWindowHeight: jobsWindowBounds.height,
    };
    safeLocalStorageSet(RUN_SETTINGS_STORAGE_KEY, JSON.stringify(nextSettings));
  }, [
    aspectRatioPreset,
    codec,
    container,
    crf,
    customAspectHeightInput,
    customAspectWidthInput,
    interpolationTargetFps,
    isBlindPanelOpen,
    isCleanupPanelOpen,
    isInputPanelOpen,
    isInterpolationStepEnabled,
    isOutputPanelOpen,
    isUpscaleStepEnabled,
    jobsWindowBounds.height,
    jobsWindowBounds.left,
    jobsWindowBounds.top,
    jobsWindowBounds.width,
    modelId,
    outputMode,
    previewDurationInput,
    previewMode,
    pytorchRunner,
    qualityPreset,
    resolutionBasis,
    segmentDurationInput,
    selectedGpuId,
    targetHeightInput,
    targetWidthInput,
    tileSize,
  ]);

  useEffect(() => {
    if (!isCleanupPanelOpen || !jobsPanelRef.current || typeof ResizeObserver === "undefined") {
      return undefined;
    }

    const panel = jobsPanelRef.current;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }

      const nextWidth = Math.round(entry.contentRect.width);
      const nextHeight = Math.round(entry.contentRect.height);
      setJobsWindowBounds((current) => {
        const normalized = clampJobsWindowBounds(current, nextWidth, nextHeight);
        if (
          normalized.left === current.left
          && normalized.top === current.top
          && normalized.width === current.width
          && normalized.height === current.height
        ) {
          return current;
        }
        return normalized;
      });
    });
    observer.observe(panel);
    return () => observer.disconnect();
  }, [isCleanupPanelOpen]);

  useEffect(() => {
    if (!jobsWindowDragState) {
      return undefined;
    }

    const handleMouseMove = (event: MouseEvent) => {
      setJobsWindowBounds((current) => {
        const nextLeft = event.clientX - jobsWindowDragState.pointerOffsetX;
        const nextTop = event.clientY - jobsWindowDragState.pointerOffsetY;
        return clampJobsWindowBounds({ ...current, left: nextLeft, top: nextTop }, current.width, current.height);
      });
    };

    const handleMouseUp = () => {
      setJobsWindowDragState(null);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [jobsWindowDragState]);

  useEffect(() => {
    const handleResize = () => {
      setJobsWindowBounds((current) => clampJobsWindowBounds(current, current.width, current.height));
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    setExpandedCleanupJobIds((current) => current.filter((jobId) => cleanupJobs.some((job) => job.id === jobId)));
  }, [cleanupJobs]);

  async function ensureRuntime(): Promise<RuntimeStatus> {
    setStatus("Preparing runtime assets...");
    const nextRuntime = await desktopApi.ensureRuntimeAssets();
    setRuntime(nextRuntime);
    return nextRuntime;
  }

  async function loadSourceFromPath(sourcePath: string): Promise<SourceVideoSummary> {
    await ensureRuntime();
    const summary = await desktopApi.probeSourceVideo(sourcePath);
    setSource(summary);
    await resolveSourcePreviewUrl(summary.previewPath || summary.path);
    setPreviewPlaybackPath(null);
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
    setIsComparisonWorkspaceOpen(false);
    setComparisonCurrentTime(0);
    setComparisonDuration(0);
    setComparisonPlaying(false);
    return summary;
  }

  function resolveRepeatRequest(job: TrackedJobEntry): ResolvedRepeatRequest | null {
    if (job.jobKind !== "pipeline") {
      return null;
    }

    if (job.pipelineRunDetails?.request) {
      const exactRequest = job.pipelineRunDetails.request;
      return {
        request: {
          ...exactRequest,
          sourcePath: job.sourcePath ?? exactRequest.sourcePath,
          modelId: job.modelId ?? exactRequest.modelId,
          codec: (job.codec === "h264" || job.codec === "h265") ? job.codec : exactRequest.codec,
          container: (job.container === "mp4" || job.container === "mkv") ? job.container : exactRequest.container,
          outputPath: normalizeOutputPath(job.outputPath ?? exactRequest.outputPath, (job.container === "mp4" || job.container === "mkv") ? job.container : exactRequest.container),
        },
        exact: true,
      };
    }

    if (!job.sourcePath || !job.outputPath || !job.modelId || (job.codec !== "h264" && job.codec !== "h265") || (job.container !== "mp4" && job.container !== "mkv")) {
      return null;
    }

    const inferredInterpolationMode: InterpolationMode = (job.progress.interpolatedFrames ?? 0) > 0
      ? (job.progress.upscaledFrames ?? 0) > 0
        ? "afterUpscale"
        : "interpolateOnly"
      : "off";
    const historicalBackend = getBackendDefinition(getModelDefinition(job.modelId).backendId);

    return {
      exact: false,
      request: {
        sourcePath: job.sourcePath,
        modelId: job.modelId,
        outputMode,
        qualityPreset,
        interpolationMode: inferredInterpolationMode,
        interpolationTargetFps: inferredInterpolationMode === "off" ? null : interpolationTargetFps,
        pytorchRunner: historicalBackend.id === "pytorch-image-sr" ? recommendedPytorchRunner(job.modelId) : "torch",
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
        previewMode: false,
        previewDurationSeconds: null,
        segmentDurationSeconds: segmentDurationSeconds ?? 10,
        outputPath: normalizeOutputPath(job.outputPath, job.container),
        codec: job.codec,
        container: job.container,
        tileSize,
        fp16: false,
        crf,
      },
    };
  }

  async function applyRepeatedPipelineRequest(request: RealesrganJobRequest): Promise<void> {
    const nextInterpolationEnabled = request.interpolationMode !== "off";
    const nextUpscaleEnabled = request.interpolationMode !== "interpolateOnly";

    setModelId(request.modelId);
    setOutputMode(request.outputMode);
    setQualityPreset(request.qualityPreset);
    setSelectedGpuId(request.gpuId);
    setAspectRatioPreset(request.aspectRatioPreset);
    setCustomAspectWidthInput(request.customAspectWidth ? String(request.customAspectWidth) : "");
    setCustomAspectHeightInput(request.customAspectHeight ? String(request.customAspectHeight) : "");
    setResolutionBasis(request.resolutionBasis);
    setTargetWidthInput(request.targetWidth ? String(request.targetWidth) : "");
    setTargetHeightInput(request.targetHeight ? String(request.targetHeight) : "");
    setCodec(request.codec);
    setContainer(request.container);
    setTileSize(request.tileSize);
    setCrf(request.crf);
    setIsUpscaleStepEnabled(nextUpscaleEnabled);
    setIsInterpolationStepEnabled(nextInterpolationEnabled);
    setInterpolationTargetFps(request.interpolationTargetFps ?? 60);
    setPytorchRunner(request.pytorchRunner);
    setPreviewMode(request.previewMode);
    setPreviewDurationInput(request.previewDurationSeconds ? String(request.previewDurationSeconds) : "8");
    setSegmentDurationInput(request.segmentDurationSeconds ? String(request.segmentDurationSeconds) : "10");
    setCropRect(
      request.cropLeft !== null && request.cropTop !== null && request.cropWidth !== null && request.cropHeight !== null
        ? {
            left: request.cropLeft,
            top: request.cropTop,
            width: request.cropWidth,
            height: request.cropHeight,
          }
        : null
    );

    const summary = await loadSourceFromPath(request.sourcePath);
  setBlindComparisonStartOffsetSeconds(normalizePreviewStartOffsetSeconds(summary, request.previewStartOffsetSeconds ?? 0));
    setOutputPath(normalizeOutputPath(request.outputPath, request.container));

    if (!supportsEmbeddedFullLengthPreview(summary.container)) {
      if (summary.durationSeconds <= AUTO_PREVIEW_UPGRADE_MAX_DURATION_SECONDS) {
        setStatus("Run settings restored. Preparing full-length preview in the background...");
        void startSourceConversionJob(summary.path, "preview", { background: true });
      } else {
        setStatus(`Run settings restored from ${pathLeaf(request.sourcePath)}.`);
      }
    } else {
      setStatus(`Run settings restored from ${pathLeaf(request.sourcePath)}.`);
    }
  }

  async function repeatTrackedJob(job: TrackedJobEntry): Promise<void> {
    const resolvedRepeat = resolveRepeatRequest(job);
    if (!resolvedRepeat) {
      setError("This job does not contain enough recorded information to restore template settings.");
      setStatus("Template settings are unavailable for the selected job.");
      return;
    }

    if (!resolvedRepeat.exact) {
      setStatus(`Restoring ${pathLeaf(resolvedRepeat.request.sourcePath)} with recorded source/output settings and current advanced defaults.`);
    }

    await repeatPipelineRun(resolvedRepeat.request);
  }

  async function handoffPipelineRequestToMainWindow(request: RealesrganJobRequest, action: RepeatPipelineRequestAction): Promise<void> {
    safeLocalStorageSet(REPEAT_PIPELINE_REQUEST_STORAGE_KEY, JSON.stringify(buildRepeatPipelineRequestEnvelope(request, Date.now(), action)));
    try {
      const { WebviewWindow } = await import("@tauri-apps/api/webviewWindow");
      const mainWindow = await WebviewWindow.getByLabel("main");
      if (mainWindow) {
        await mainWindow.show();
        await mainWindow.setFocus();
      }
    } catch {
      // Ignore focus handoff failures; the replay request is already persisted.
    }
  }

  async function restartTrackedJob(job: TrackedJobEntry): Promise<void> {
    const resolvedRepeat = resolveRepeatRequest(job);
    if (!resolvedRepeat) {
      setError("This job does not contain enough recorded information to restart.");
      setStatus("Restart is unavailable for the selected job.");
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      if (isJobsOnlyView) {
        await handoffPipelineRequestToMainWindow(resolvedRepeat.request, "restart");
        setStatus(`Sent ${pathLeaf(resolvedRepeat.request.sourcePath)} to the main window for restart.`);
        return;
      }

      setStatus(`Reloading ${pathLeaf(resolvedRepeat.request.sourcePath)} for restart...`);
      await applyRepeatedPipelineRequest(resolvedRepeat.request);
      setStatus(`Restarting ${pathLeaf(resolvedRepeat.request.sourcePath)}...`);
      await startPipelineFromRequest(resolvedRepeat.request, { ensureRuntime: false, queuedStatus: "Restarted job queued." });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to restart the selected job.");
    } finally {
      setIsBusy(false);
    }
  }

  async function repeatPipelineRun(request: RealesrganJobRequest): Promise<void> {
    try {
      setIsBusy(true);
      setError(null);

      if (isJobsOnlyView) {
        await handoffPipelineRequestToMainWindow(request, "repeat");
        setStatus(`Sent ${pathLeaf(request.sourcePath)} template settings to the main window.`);
        return;
      }

      await applyRepeatedPipelineRequest(request);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to restore template settings from the selected run.");
    } finally {
      setIsBusy(false);
    }
  }

  function comparisonVideoTargets(): HTMLVideoElement[] {
    return [
      comparisonSourceVideoRef.current,
      ...comparisonEntries.map((entry) => comparisonSampleVideoRefs.current[entry.sampleId] ?? null),
    ].filter((video): video is HTMLVideoElement => Boolean(video));
  }

  function setComparisonSourceVideoRef(node: HTMLVideoElement | null): void {
    comparisonSourceVideoRef.current = node;
  }

  function setComparisonSampleVideoRef(sampleId: string, node: HTMLVideoElement | null): void {
    if (node) {
      comparisonSampleVideoRefs.current[sampleId] = node;
    } else {
      delete comparisonSampleVideoRefs.current[sampleId];
    }
  }

  function updateComparisonFocusFromEventTarget(target: EventTarget | null, clientX: number, clientY: number): void {
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const rect = target.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }

    const nextFocusX = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    const nextFocusY = Math.max(0, Math.min(100, ((clientY - rect.top) / rect.height) * 100));
    setComparisonFocusPresetId("manual");
    setComparisonFocusX(nextFocusX);
    setComparisonFocusY(nextFocusY);
  }

  function handleComparisonViewportPointerDown(event: ReactPointerEvent<HTMLElement>): void {
    updateComparisonFocusFromEventTarget(event.currentTarget, event.clientX, event.clientY);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handleComparisonViewportPointerMove(event: ReactPointerEvent<HTMLElement>): void {
    if ((event.buttons & 1) !== 1) {
      return;
    }
    updateComparisonFocusFromEventTarget(event.currentTarget, event.clientX, event.clientY);
  }

  async function ensureComparisonVideoReady(video: HTMLVideoElement): Promise<boolean> {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      return true;
    }

    return await new Promise<boolean>((resolve) => {
      let settled = false;
      const complete = (ready: boolean) => {
        if (settled) {
          return;
        }
        settled = true;
        window.clearTimeout(timeoutId);
        video.removeEventListener("loadeddata", handleReady);
        video.removeEventListener("canplay", handleReady);
        video.removeEventListener("error", handleError);
        resolve(ready);
      };

      const handleReady = () => complete(true);
      const handleError = () => complete(false);
      const timeoutId = window.setTimeout(() => complete(video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA), 5_000);

      video.addEventListener("loadeddata", handleReady, { once: true });
      video.addEventListener("canplay", handleReady, { once: true });
      video.addEventListener("error", handleError, { once: true });
      video.load();
    });
  }

  function openComparisonWorkspace(): void {
    if (isComparisonWorkspaceOpen) {
      return;
    }
    setIsComparisonWorkspaceOpen(true);
  }

  function closeComparisonWorkspace(): void {
    comparisonVideoTargets().forEach((video) => video.pause());
    setComparisonPlaying(false);
    comparisonDesiredTimeRef.current = 0;
    setIsComparisonWorkspaceOpen(false);
  }

  function refreshComparisonDuration(): void {
    const durations = comparisonVideoTargets()
      .map((video) => video.duration)
      .filter((value): value is number => Number.isFinite(value ?? NaN));
    setComparisonDuration(durations.length > 0 ? Math.min(...durations) : 0);
  }

  function syncComparisonTime(nextTime: number): void {
    const boundedTime = comparisonDuration > 0 ? Math.min(nextTime, comparisonDuration) : nextTime;
    comparisonDesiredTimeRef.current = boundedTime;
    for (const video of comparisonVideoTargets()) {
      if (!video || !Number.isFinite(video.duration)) {
        continue;
      }

      const clamped = Math.max(0, Math.min(boundedTime, video.duration || boundedTime));
      if (Math.abs(video.currentTime - clamped) > 0.05) {
        video.currentTime = clamped;
      }
    }
    setComparisonCurrentTime(boundedTime);
  }

  function syncComparisonFrame(nextFrame: number): void {
    if (comparisonFrameRate <= 0) {
      syncComparisonTime(nextFrame);
      return;
    }

    const clampedFrame = Math.max(0, Math.min(nextFrame, comparisonTimelineMax));
    syncComparisonTime(clampedFrame / comparisonFrameRate);
  }

  function handleComparisonTimelineInput(nextValue: number): void {
    if (comparisonFrameCount > 0) {
      syncComparisonFrame(nextValue);
      return;
    }
    syncComparisonTime(nextValue);
  }

  function seekComparisonTimelineFromPointer(event: ReactPointerEvent<HTMLInputElement>): void {
    const input = event.currentTarget;
    const bounds = input.getBoundingClientRect();
    if (bounds.width <= 0) {
      return;
    }

    const min = Number(input.min || "0");
    const max = Number(input.max || "0");
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
      return;
    }

    const ratio = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    const rawValue = min + ((max - min) * ratio);
    const nextValue = comparisonFrameCount > 0 ? Math.round(rawValue) : rawValue;
    handleComparisonTimelineInput(nextValue);
  }

  function handleComparisonLoadedMetadata(): void {
    refreshComparisonDuration();
    if (comparisonFrameRate > 0 && comparisonCurrentFrame > 0) {
      syncComparisonFrame(comparisonCurrentFrame);
      return;
    }
    if (comparisonCurrentTime > 0) {
      syncComparisonTime(comparisonCurrentTime);
    }
  }

  function handleComparisonSourceTimeUpdate(): void {
    const sourceVideo = comparisonSourceVideoRef.current;
    if (!sourceVideo) {
      return;
    }

    const rawNextTime = sourceVideo.currentTime;
    const nextTime = comparisonDuration > 0 ? Math.min(rawNextTime, comparisonDuration) : rawNextTime;
    if (!comparisonPlaying && Math.abs(nextTime - comparisonDesiredTimeRef.current) > 0.08) {
      return;
    }

    comparisonDesiredTimeRef.current = nextTime;
    setComparisonCurrentTime(nextTime);
    for (const video of comparisonVideoTargets()) {
      if (video === sourceVideo || !Number.isFinite(video.duration)) {
        continue;
      }

      if (Math.abs(video.currentTime - nextTime) > 0.08) {
        video.currentTime = Math.max(0, Math.min(nextTime, video.duration || nextTime));
      }
    }

    if (comparisonDuration > 0 && rawNextTime >= comparisonDuration - 0.02) {
      comparisonVideoTargets().forEach((video) => video.pause());
      setComparisonPlaying(false);
    }
  }

  async function toggleComparisonPlayback(): Promise<void> {
    const videos = comparisonVideoTargets();
    if (videos.length < 2) {
      return;
    }

    if (comparisonPlaying) {
      videos.forEach((video) => video.pause());
      setComparisonPlaying(false);
      return;
    }

    const targetTime = comparisonFrameRate > 0
      ? (comparisonCurrentFrame >= comparisonTimelineMax ? 0 : comparisonCurrentFrame / comparisonFrameRate)
      : (comparisonDuration > 0 && comparisonCurrentTime >= comparisonDuration - 0.05 ? 0 : comparisonCurrentTime);

    const readyVideos = (await Promise.all(videos.map(async (video) => ({
      video,
      ready: await ensureComparisonVideoReady(video),
    })))).filter((entry) => entry.ready).map((entry) => entry.video);

    if (readyVideos.length === 0) {
      setComparisonPlaying(false);
      setStatus("Comparison media is still loading.");
      return;
    }

    const playResults = await Promise.allSettled(readyVideos.map(async (video) => {
      if (Number.isFinite(video.duration)) {
        const clamped = Math.max(0, Math.min(targetTime, video.duration || targetTime));
        if (Math.abs(video.currentTime - clamped) > 0.05) {
          video.currentTime = clamped;
        }
      }
      await video.play();
    }));

    const playedCount = playResults.filter((result) => result.status === "fulfilled").length;
    setComparisonPlaying(playedCount > 0);
    if (playedCount === 0) {
      setStatus("Comparison media is still loading.");
    }
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

  function captureBlindComparisonStartFromPreview(): void {
    setBlindComparisonStartOffsetSeconds(normalizePreviewStartOffsetSeconds(source, sourcePreviewCurrentTime));
  }

  function resetBlindComparisonStartOffset(): void {
    setBlindComparisonStartOffsetSeconds(0);
  }

  function jumpPreviewToBlindComparisonStart(): void {
    seekSourcePreview(normalizedBlindComparisonStartOffsetSeconds);
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

  function buildPipelineRequest(
    targetModelId: ModelId,
    targetOutputPath: string,
    quickPreview: boolean,
    quickPreviewSeconds: number | null,
    overrides?: Partial<Pick<RealesrganJobRequest, "codec" | "container" | "previewStartOffsetSeconds">>,
  ): RealesrganJobRequest {
    if (!source) {
      throw new Error("Select a source video before starting a pipeline.");
    }

    const resolvedCodec = overrides?.codec ?? codec;
    const resolvedContainer = overrides?.container ?? container;
  const resolvedPreviewStartOffsetSeconds = quickPreview ? (overrides?.previewStartOffsetSeconds ?? null) : null;

    return {
      sourcePath: source.path,
      modelId: targetModelId,
      outputMode,
      qualityPreset,
      interpolationMode,
      interpolationTargetFps: interpolationEnabled ? interpolationTargetFps : null,
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
      previewStartOffsetSeconds: resolvedPreviewStartOffsetSeconds,
      segmentDurationSeconds: quickPreview ? null : segmentDurationSeconds,
      outputPath: targetOutputPath,
      codec: resolvedCodec,
      container: resolvedContainer,
      tileSize,
      fp16: false,
      crf,
    };
  }

  async function startPipelineFromRequest(request: RealesrganJobRequest, options?: { ensureRuntime?: boolean; queuedStatus?: string }): Promise<string> {
    const nextRuntime = options?.ensureRuntime === false
      ? (runtime ?? await ensureRuntime())
      : await ensureRuntime();

    const requestModel = getModelDefinition(request.modelId);
    const launchRequirement = modelLaunchRequirement(requestModel, nextRuntime);
    if (launchRequirement) {
      throw new Error(launchRequirement);
    }

    setActivePipelineRequest(request);
    setOutputPath(normalizeOutputPath(request.outputPath, request.container));
    setIsPipelineLaunchPending(true);
    const jobId = await desktopApi.startPipeline(request);
    setResult(null);
    setPipelineProgressEvents([]);
    setLastPipelineProgressAt(null);
    pipelineProgressSignatureRef.current = null;
    setIsPipelineLaunchPending(false);
    setPipelineJob(createQueuedJob(jobId));
    setActiveJobId(jobId);
    setStatus(options?.queuedStatus ?? "Job queued.");
    return jobId;
  }

  async function confirmInterpolationPolicy(): Promise<boolean> {
    const message = buildInterpolationWarning(source, interpolationMode, interpolationTargetFps);
    if (!message) {
      return true;
    }
    return window.confirm(message);
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
      setIsComparisonWorkspaceOpen(false);
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

    if (!isSelectedModelLaunchable) {
      setError(selectedModelLaunchRequirement);
      setStatus("Selected model needs additional runtime setup before it can run.");
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      if (!(await confirmInterpolationPolicy())) {
        setStatus("Interpolation start cancelled.");
        return;
      }
      const selectedOutputPath = outputPath ?? await chooseOutputFile();
      if (!selectedOutputPath) {
        setStatus("Output selection cancelled.");
        return;
      }

      setStatus("Starting Real-ESRGAN pipeline...");
      await startPipelineFromRequest(
        buildPipelineRequest(modelId, selectedOutputPath, previewMode, previewMode ? previewDurationSeconds : null),
        { ensureRuntime: true, queuedStatus: "Job queued." }
      );
    } catch (caught) {
      setIsPipelineLaunchPending(false);
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

  async function pausePipeline(): Promise<void> {
    if (!activeJobId) {
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await desktopApi.pausePipelineJob(activeJobId);
      setStatus("Pausing pipeline...");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to pause pipeline.");
    } finally {
      setIsBusy(false);
    }
  }

  async function resumePipeline(): Promise<void> {
    if (!activeJobId) {
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await desktopApi.resumePipelineJob(activeJobId);
      setStatus("Resuming pipeline...");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to resume pipeline.");
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

  async function pauseSourceConversion(): Promise<void> {
    if (!sourceConversionJobId) {
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await desktopApi.pauseSourceConversionJob(sourceConversionJobId);
      setStatus("Pausing source conversion...");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to pause source conversion.");
    } finally {
      setIsBusy(false);
    }
  }

  async function resumeSourceConversion(): Promise<void> {
    if (!sourceConversionJobId) {
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await desktopApi.resumeSourceConversionJob(sourceConversionJobId);
      setStatus("Resuming source conversion...");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Failed to resume source conversion.");
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

  async function deleteManagedArtifact(path: string, label: string, afterDelete: () => void, confirmationDetails: string[] = []): Promise<void> {
    const confirmed = window.confirm(buildDeleteConfirmation(`Delete ${label}?`, path, confirmationDetails));
    if (!confirmed) {
      setStatus(`${label} deletion cancelled.`);
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await desktopApi.deleteManagedPath(path);
      afterDelete();
      await refreshManagedWorkspaceState();
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
      await refreshManagedWorkspaceState();
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
    }, [
      "Removes every managed artifact currently stored in this scratch pool.",
      "Active jobs must already be stopped before the pool can be cleared.",
    ]);
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

  function toggleBlindComparisonModel(modelId: ModelId): void {
    setSelectedBlindComparisonModelIds((current) => toggleIncludedModel(current, modelId));
  }

  function restoreRecommendedBlindComparisonModels(): void {
    setSelectedBlindComparisonModelIds(blindComparisonDefaultCandidates.map((candidate) => candidate.value));
  }

  function selectAllBlindComparisonModels(): void {
    setSelectedBlindComparisonModelIds(blindComparisonAvailableModels.map((candidate) => candidate.value));
  }

  async function runBlindComparison(): Promise<void> {
    if (!source) {
      setError("Select a source video before starting blind comparison.");
      return;
    }

    if (selectedBlindComparisonModelIds.length < 2) {
      setError("Blind comparison needs at least two runnable models.");
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      await ensureRuntime();
      const duration = previewDurationSeconds ?? 8;
      const startOffsetSeconds = normalizePreviewStartOffsetSeconds(source, blindComparisonStartOffsetSeconds);
      const shuffledModels = shuffleModels(selectedBlindComparisonModelIds);
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
        previewStartOffsetSeconds: startOffsetSeconds,
        selectedSampleId: null,
        winnerModelId: null,
        revealed: false,
        error: null,
      });
      setIsComparisonWorkspaceOpen(false);
      comparisonSampleVideoRefs.current = {};
      setComparisonCurrentTime(0);
      setComparisonDuration(0);
      setComparisonPlaying(false);
      setStatus("Blind comparison queued.");

      void (async () => {
        for (const entry of startedEntries) {
          try {
            const outputPath = blindComparisonOutputPath(source, BLIND_COMPARISON_PREVIEW_CONTAINER, entry.modelId, entry.anonymousLabel, runToken);
            const jobId = await desktopApi.startPipeline(
              buildPipelineRequest(entry.modelId, outputPath, true, duration, {
                codec: BLIND_COMPARISON_PREVIEW_CODEC,
                container: BLIND_COMPARISON_PREVIEW_CONTAINER,
                previewStartOffsetSeconds: startOffsetSeconds,
              })
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
        previewStartOffsetSeconds: blindComparison.previewStartOffsetSeconds,
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

  function matchInputFormat(): void {
    if (!source) {
      return;
    }

    if (matchedInputContainer) {
      updateContainer(matchedInputContainer);
    }

    if (matchedInputCodec) {
      setCodec(matchedInputCodec);
    }
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
      {!isJobsOnlyView ? (
      <>
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Windows-first video upgrade workbench</p>
          <h1>{APP_NAME}</h1>
          <p className="summary">
            Compare spatial upscaling results, inspect 4K framing behavior, and prepare for
            frame-rate upgrades from the same desktop workflow before committing to a full export.
          </p>
        </div>
        <div className="status-card status-card-compact" data-testid="top-status-panel">
          <div className="status-card-content">
            <div className="status-card-header">
              <div className="status-card-header-copy">
                <span className="status-label">{compactStatusTitle}</span>
                <div className="status-card-headline-row">
                  <strong>{isSourceConversionRunning ? "Preparing source preview" : activeManagedJob ? managedJobLabel(activeManagedJob.label, activeManagedJob.sourcePath) : compactPipelineLabel}</strong>
                </div>
              </div>
            </div>
            <span className="status-primary-text">{progressMessage}</span>
            <div className="progress-shell" aria-label="Pipeline progress">
              <div className="progress-bar" style={{ width: `${progressPercent}%` }} />
            </div>
            <div className="status-metric-row">
              <span data-testid="top-status-percent">{progressPercent}% complete</span>
              <span data-testid="top-status-eta">
                {activePrimaryJob && (activePrimaryJob.progress.estimatedRemainingSeconds ?? 0) > 0
                  ? `ETA ${formatElapsedSeconds(activePrimaryJob.progress.estimatedRemainingSeconds)}`
                  : isPipelineRunning || isSourceConversionRunning
                    ? "ETA pending"
                    : "Not running"}
              </span>
            </div>
            <div className="status-secondary-row">
              <span className="status-secondary-text" data-testid="pipeline-launch-state" data-state={pipelineLaunchState}>Pipeline launch {pipelineLaunchStateLabel}</span>
            </div>
            {pipelineJob ? (
              <div className="status-phase-stack">
                {compactPhaseBars
                  .filter((entry) => entry.id === "upscale" || interpolationEnabled)
                  .map((entry) => (
                    <div key={entry.id} className="status-phase-row">
                      <span>{entry.label}</span>
                      <div className="progress-shell status-mini-progress" aria-hidden="true">
                        <div className="progress-bar" style={{ width: `${entry.value * 100}%` }} />
                      </div>
                      <span>{entry.summary}</span>
                    </div>
                  ))}
              </div>
            ) : null}
            <div className="status-secondary-row">
              <span className="status-secondary-text">{compactStatusDetail}</span>
            </div>
          </div>
          <div className="status-card-rail">
            <button
              type="button"
              className="action-button secondary-button jobs-launch-button"
              data-testid="job-cleanup-panel-toggle"
              onClick={() => void openJobsWindow()}
              title={canOpenNativeJobsWindow ? `Open jobs in a separate native window. ${cleanupJobs.length} tracked jobs.` : `Show the jobs workspace in this browser session. ${cleanupJobs.length} tracked jobs.`}
            >
              {isCleanupPanelOpen && !canOpenNativeJobsWindow ? `Hide Jobs [${cleanupJobs.length}]` : `Jobs [${cleanupJobs.length}]`}
            </button>
            {topStatusPauseAction || topStatusStopAction ? (
              <div className="status-control-row">
                {topStatusPauseAction ? (
                  <button
                    type="button"
                    className="action-button status-icon-button status-pause-button"
                    data-testid="top-status-pause-button"
                    onClick={topStatusPauseAction}
                    disabled={isBusy}
                    aria-label={isPipelinePaused || isSourceConversionPaused ? "Resume the active job" : "Pause the active job"}
                    title={isPipelinePaused || isSourceConversionPaused ? "Resume the active job." : "Pause the active job."}
                  >
                    {isPipelinePaused || isSourceConversionPaused ? <span className="status-resume-icon" aria-hidden="true" /> : <span className="status-pause-icon" aria-hidden="true" />}
                  </button>
                ) : null}
                {topStatusStopAction ? (
                  <button
                    type="button"
                    className="action-button status-icon-button status-stop-button"
                    data-testid="top-status-stop-button"
                    onClick={topStatusStopAction}
                    disabled={isBusy}
                    aria-label={isPipelineRunning ? "Stop the active job" : "Stop the active source conversion"}
                    title={isPipelineRunning ? "Stop the active job." : "Stop the active source conversion."}
                  >
                    <span className="status-stop-icon" aria-hidden="true" />
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <section className="workspace-grid primary-panel-grid">
        <div className="workspace-column">
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
              <div className="source-metadata-stack">
                <dl className="facts">
                  <div><dt>Path</dt><dd>{source.path}</dd></div>
                  <div><dt>Preview</dt><dd data-testid="source-preview-mode">{previewUpgradeAvailable ? "Full-length converted preview" : usingFallbackPreviewClip ? "Short fallback preview clip" : "Direct source playback"}</dd></div>
                  <div><dt>Input Size</dt><dd>{formatBytes(sourcePathStats?.sizeBytes ?? 0)}</dd></div>
                  <div><dt>Container</dt><dd>{source.container.toUpperCase()}</dd></div>
                  <div><dt>Duration</dt><dd>{formatClockTime(source.durationSeconds)} ({source.durationSeconds.toFixed(2)}s)</dd></div>
                  <div><dt>Source Bitrate</dt><dd>{formatBitrateKbps(source.sourceBitrateKbps)}</dd></div>
                </dl>
                <div className="source-metadata-grid">
                  <details className="source-detail-disclosure source-metadata-card" data-testid="source-video-details">
                    <summary className="source-detail-summary" title={buildSourceVideoSummary(source)}>
                      <span className="source-detail-summary-label">Video Details</span>
                      <span className="source-detail-summary-value">{buildSourceVideoSummary(source)}</span>
                    </summary>
                    <dl className="facts compact-facts source-detail-facts">
                      <div><dt>Resolution</dt><dd>{source.width} x {source.height}</dd></div>
                      <div><dt>Aspect Ratio</dt><dd>{formatAspectRatio(source.width, source.height)}</dd></div>
                      <div><dt>Frame Rate</dt><dd>{source.frameRate.toFixed(3)} fps</dd></div>
                      <div><dt>Codec</dt><dd>{formatMediaLabel(source.videoCodec)}</dd></div>
                      <div><dt>Profile</dt><dd>{source.videoProfile?.trim() || "Unknown"}</dd></div>
                      <div><dt>Pixel Format</dt><dd>{source.pixelFormat?.trim() || "Unknown"}</dd></div>
                    </dl>
                  </details>
                  <details className="source-detail-disclosure source-metadata-card" data-testid="source-audio-details">
                    <summary className="source-detail-summary" title={buildSourceAudioSummary(source)}>
                      <span className="source-detail-summary-label">Audio Details</span>
                      <span className="source-detail-summary-value">{buildSourceAudioSummary(source)}</span>
                    </summary>
                    {source.hasAudio ? (
                      <dl className="facts compact-facts source-detail-facts">
                        <div><dt>Track</dt><dd>Present</dd></div>
                        <div><dt>Codec</dt><dd>{formatMediaLabel(source.audioCodec)}</dd></div>
                        <div><dt>Profile</dt><dd>{source.audioProfile?.trim() || "Unknown"}</dd></div>
                        <div><dt>Sample Rate</dt><dd>{formatSampleRate(source.audioSampleRate)}</dd></div>
                        <div><dt>Channels</dt><dd>{formatTitleCase(source.audioChannels)}</dd></div>
                        <div><dt>Bitrate</dt><dd>{formatBitrateKbps(source.audioBitrateKbps)}</dd></div>
                      </dl>
                    ) : (
                      <p className="summary source-detail-empty">No audio stream detected in the selected source.</p>
                    )}
                  </details>
                </div>
              </div>
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
        <article className="panel roadmap-note" data-testid="dynamic-crop-panel">
          <div className="roadmap-note-header">
            <p className="eyebrow">Roadmap</p>
            <strong>Dynamic crop is still experimental</strong>
          </div>
          <p className="summary" data-testid="dynamic-crop-summary">
            It stays out of the main workflow for now. The next version will explore object-aware crop tracking without competing with the core input and export controls.
          </p>
        </article>

        <ExpandablePanel
          title="Blind Test"
          subtitle={`${blindComparisonAvailableModels.length} available • ${selectedBlindComparisonModelIds.length} selected`}
          isOpen={isBlindPanelOpen}
          onToggle={() => setIsBlindPanelOpen((current) => !current)}
          testId="blind-test-panel"
        >
          <section className="blind-comparison-panel" data-testid="blind-comparison-panel">
            <p className="summary">
              Pick which visible runnable models to include, run anonymized {previewDurationSeconds ?? 8}s preview exports for that set,
              then compare the same frame, zoom, and focus point across every sample before the app reveals which model produced which result.
            </p>
            <div className="blind-start-offset-panel" data-testid="blind-start-offset-panel">
              <div className="blind-start-offset-summary-row">
                <strong>Comparison Start</strong>
                <span data-testid="blind-start-offset-readout">{formatClockTime(normalizedBlindComparisonStartOffsetSeconds)}</span>
              </div>
              <p className="summary blind-start-offset-note" data-testid="blind-start-offset-note">
                Scrub the source preview, then capture that position. The comparison clip begins on the first full frame at or after the selected spot.
              </p>
              <div className="blind-start-offset-actions">
                <button
                  type="button"
                  className="action-button secondary-button"
                  data-testid="blind-capture-current-preview-position"
                  onClick={captureBlindComparisonStartFromPreview}
                  disabled={!source || isBlindComparisonRunning || isBusy}
                >
                  Use Current Preview Position
                </button>
                <button
                  type="button"
                  className="action-button secondary-button"
                  data-testid="blind-jump-preview-to-start-offset"
                  onClick={jumpPreviewToBlindComparisonStart}
                  disabled={!source || isBlindComparisonRunning || isBusy}
                >
                  Jump Preview To Start
                </button>
                <button
                  type="button"
                  className="action-button secondary-button"
                  data-testid="blind-reset-start-offset"
                  onClick={resetBlindComparisonStartOffset}
                  disabled={!source || isBlindComparisonRunning || isBusy || normalizedBlindComparisonStartOffsetSeconds <= 0}
                >
                  Reset To Start
                </button>
              </div>
            </div>
            <div className="blind-model-toolbar" data-testid="blind-model-selector">
              <div className="blind-model-toolbar-actions">
                <button type="button" className="action-button secondary-button" data-testid="blind-select-all-models" onClick={selectAllBlindComparisonModels} disabled={isBlindComparisonRunning || isBusy}>
                  Select All Visible Models
                </button>
                <button type="button" className="action-button secondary-button" data-testid="blind-restore-recommended-models" onClick={restoreRecommendedBlindComparisonModels} disabled={isBlindComparisonRunning || isBusy}>
                  Restore Recommended Set
                </button>
              </div>
              <div className="blind-model-selector-grid">
                {blindComparisonAvailableModels.map((model) => {
                  const isSelected = selectedBlindComparisonModelIds.includes(model.value);
                  return (
                    <label key={model.value} className={`blind-model-option${isSelected ? " blind-model-option-selected" : ""}`} data-testid={`blind-model-option-${model.value}`}>
                      <input
                        type="checkbox"
                        data-testid={`blind-model-toggle-${model.value}`}
                        checked={isSelected}
                        disabled={isBlindComparisonRunning || isBusy}
                        onChange={() => toggleBlindComparisonModel(model.value)}
                      />
                      <span className="blind-model-option-copy">
                        <strong>{model.label}</strong>
                        <span>{model.summary}</span>
                      </span>
                    </label>
                  );
                })}
              </div>
              <p className="summary blind-model-toolbar-note" data-testid="blind-model-toolbar-note">
                The selection list shows real model names before the run. Once the comparison starts, each output stays anonymous until you pick a winner.
              </p>
            </div>
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
                  const comparisonPreviewSrc = resolvedComparisonPreviewUrls[entry.sampleId] ?? (previewPath ? desktopApi.toPreviewSrc(previewPath) : "");
                  const comparisonPreviewMimeType = previewMimeType(previewPath);
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
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            openComparisonWorkspace();
                          }}
                        >
                          {isComparisonWorkspaceOpen ? (
                            <div className="preview-shell preview-placeholder blind-placeholder" data-testid={`blind-preview-suspended-${entry.sampleId}`}>
                              <span>Comparison workspace open</span>
                            </div>
                          ) : (
                            <video key={comparisonPreviewSrc || entry.sampleId} className="result-preview clickable-preview" preload="metadata" data-testid={`blind-preview-${entry.sampleId}`} muted>
                              {comparisonPreviewSrc ? <source src={comparisonPreviewSrc} type={comparisonPreviewMimeType} /> : null}
                            </video>
                          )}
                          <span className="preview-launch-hint">Click to open the synchronized comparison workspace</span>
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
            {comparisonEntries.length > 0 && source ? (
              <section className="comparison-inspector" data-testid="comparison-inspector">
                <div className="catalog-card-header">
                  <strong>Comparison Workspace</strong>
                  <span className="catalog-chip">Synchronized across every ready sample</span>
                </div>
                <p className="summary">
                  Open the larger workspace to inspect the same frame across the source and every ready blind sample. Timeline, zoom, and focus point stay locked together.
                </p>
                <div className="inspector-sample-row comparison-ready-row">
                  {comparisonEntries.map((entry) => (
                    <span key={entry.sampleId} className="inspector-sample-button comparison-ready-pill" data-testid={`comparison-select-${entry.sampleId}`}>
                      {entry.anonymousLabel}
                    </span>
                  ))}
                </div>
                <div className="inspector-controls-grid">
                  <button type="button" className="action-button secondary-button" data-testid="open-comparison-workspace-button" onClick={openComparisonWorkspace}>
                    Open Comparison Workspace
                  </button>
                  <button type="button" className="action-button secondary-button" data-testid="open-source-external-button" onClick={() => void openMediaInDefaultApp(source.path)}>
                    Open Source Externally
                  </button>
                  <span className="summary comparison-ready-note" data-testid="comparison-ready-note">
                    {comparisonEntries.length} of {blindComparison?.entries.length ?? comparisonEntries.length} samples are ready for synchronized review.
                  </span>
                </div>
              </section>
            ) : null}
            {comparisonEntries.length > 0 && source && isComparisonWorkspaceOpen ? (
              <div className="comparison-workspace-overlay" data-testid="comparison-workspace-modal" aria-label="Comparison workspace" aria-modal="true" role="dialog">
                <div className="comparison-workspace-shell">
                  <div className="comparison-workspace-header">
                    <div className="catalog-card-header">
                      <strong>Comparison Workspace</strong>
                      <span className="catalog-chip">Source plus {comparisonEntries.length} blind samples</span>
                    </div>
                    <button type="button" className="action-button secondary-button" data-testid="comparison-workspace-close" onClick={closeComparisonWorkspace}>
                      Close Workspace
                    </button>
                  </div>
                  <div className="inspector-controls-grid comparison-workspace-controls">
                    <button type="button" className="action-button secondary-button" data-testid="comparison-play-toggle" onClick={() => void toggleComparisonPlayback()}>
                      {comparisonPlaying ? "Pause Comparison" : "Play Comparison"}
                    </button>
                    <button type="button" className="action-button secondary-button" data-testid="comparison-restart-button" onClick={restartComparisonPlayback}>
                      Restart Comparison
                    </button>
                    <button type="button" className="action-button secondary-button" data-testid="open-source-external-button" onClick={() => void openMediaInDefaultApp(source.path)}>
                      Open Source Externally
                    </button>
                  </div>
                  <label>
                    Comparison Timeline
                    <span className="summary comparison-ready-note" data-testid="comparison-timeline-readout">
                      {comparisonFrameCount > 0
                        ? `Frame ${comparisonCurrentFrame.toLocaleString()} / ${comparisonTimelineMax.toLocaleString()}`
                        : `${comparisonCurrentTime.toFixed(2)}s`}
                    </span>
                    <input
                      data-testid="comparison-time-slider"
                      type="range"
                      min={0}
                      max={comparisonFrameCount > 0 ? comparisonTimelineMax : Math.max(comparisonDuration, 0.01)}
                      step={comparisonFrameCount > 0 ? 1 : 0.01}
                      value={comparisonFrameCount > 0 ? comparisonCurrentFrame : Math.min(comparisonCurrentTime, Math.max(comparisonDuration, 0.01))}
                      onPointerDown={seekComparisonTimelineFromPointer}
                      onInput={(event) => handleComparisonTimelineInput(Number((event.target as HTMLInputElement).value))}
                      onChange={(event) => handleComparisonTimelineInput(Number(event.target.value))}
                    />
                  </label>
                  <div className="comparison-toolbar-grid">
                    <label>
                      Zoom
                      <input data-testid="comparison-zoom-slider" type="range" min={1} max={16} step={0.25} value={comparisonZoom} onChange={(event) => setComparisonZoom(Number(event.target.value))} />
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
                    {selectedComparisonPreset?.hint ?? "Move around the frame and inspect every sample against the source."}
                  </p>
                  <div className="comparison-inspector-grid comparison-workspace-grid">
                    <article className="comparison-inspector-card comparison-workspace-card">
                      <div className="catalog-card-header">
                        <strong>Source</strong>
                        <span className="catalog-chip">Reference</span>
                      </div>
                      <div
                        className="inspection-viewport comparison-workspace-viewport"
                        data-testid="comparison-source-viewport"
                        onPointerDown={handleComparisonViewportPointerDown}
                        onPointerMove={handleComparisonViewportPointerMove}
                      >
                        <video
                          key={comparisonSourcePreviewSrc || comparisonSourcePreviewPath}
                          ref={setComparisonSourceVideoRef}
                          className="inspection-video"
                          muted
                          playsInline
                          preload="auto"
                          onLoadedMetadata={handleComparisonLoadedMetadata}
                          onLoadedData={handleComparisonLoadedMetadata}
                          onTimeUpdate={handleComparisonSourceTimeUpdate}
                          onPause={() => setComparisonPlaying(false)}
                          onPlay={() => setComparisonPlaying(true)}
                          style={{ transform: `scale(${comparisonZoom})`, transformOrigin: `${comparisonFocusX}% ${comparisonFocusY}%` }}
                        >
                          {comparisonSourcePreviewSrc ? <source src={comparisonSourcePreviewSrc} type={comparisonSourcePreviewMimeType} /> : null}
                        </video>
                        <span className="inspection-crosshair" />
                      </div>
                    </article>
                    {comparisonEntries.map((entry) => {
                      const actualModel = getModelDefinition(entry.modelId);
                      const actualBackend = getBackendDefinition(actualModel.backendId);
                      const isWinner = blindComparison?.selectedSampleId === entry.sampleId;
                      const previewPath = entry.status.result?.outputPath ?? "";
                      const comparisonPreviewSrc = resolvedComparisonPreviewUrls[entry.sampleId] ?? desktopApi.toPreviewSrc(previewPath);
                      const comparisonPreviewMimeType = previewMimeType(previewPath);
                      return (
                        <article key={entry.sampleId} className="comparison-inspector-card comparison-workspace-card" data-testid={`comparison-workspace-card-${entry.sampleId}`}>
                          <div className="catalog-card-header">
                            <strong>{entry.anonymousLabel}</strong>
                            <span className="catalog-chip">Blind sample</span>
                          </div>
                          <div
                            className="inspection-viewport comparison-workspace-viewport"
                            data-testid={`comparison-sample-viewport-${entry.sampleId}`}
                            onPointerDown={handleComparisonViewportPointerDown}
                            onPointerMove={handleComparisonViewportPointerMove}
                          >
                            <video
                              key={comparisonPreviewSrc || entry.sampleId}
                              ref={(node) => setComparisonSampleVideoRef(entry.sampleId, node)}
                              className="inspection-video"
                              muted
                              playsInline
                              preload="auto"
                              onLoadedMetadata={handleComparisonLoadedMetadata}
                              onLoadedData={handleComparisonLoadedMetadata}
                              style={{ transform: `scale(${comparisonZoom})`, transformOrigin: `${comparisonFocusX}% ${comparisonFocusY}%` }}
                            >
                              {comparisonPreviewSrc ? <source src={comparisonPreviewSrc} type={comparisonPreviewMimeType} /> : null}
                            </video>
                            <span className="inspection-crosshair" />
                          </div>
                          <div className="comparison-workspace-card-actions">
                            <button type="button" className="action-button secondary-button" data-testid={`comparison-open-external-${entry.sampleId}`} onClick={() => void openMediaInDefaultApp(entry.status.result?.outputPath ?? "") }>
                              Open Externally
                            </button>
                            {blindComparison?.state === "ready" && !blindComparison.revealed ? (
                              <button type="button" className="action-button secondary-button" data-testid={`comparison-pick-${entry.sampleId}`} onClick={() => void chooseBlindWinner(entry.sampleId)}>
                                Pick {entry.anonymousLabel}
                              </button>
                            ) : null}
                            {blindComparison?.revealed ? (
                              <div className="blind-reveal-block" data-testid={`comparison-workspace-reveal-${entry.sampleId}`}>
                                <strong>{actualModel.label}</strong>
                                <span>{actualBackend.label}</span>
                                {isWinner ? <span className="winner-pill">Selected winner</span> : null}
                              </div>
                            ) : null}
                          </div>
                        </article>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : null}
            {blindComparison?.error ? <p className="error-text">{blindComparison.error}</p> : null}
          </section>
        </ExpandablePanel>
        </div>

        <div className="workspace-column">

        <ExpandablePanel
          title="Pipeline"
          subtitle={result ? "Output ready" : isPipelineRunning ? "Running now" : "Configure and run"}
          isOpen={isOutputPanelOpen}
          onToggle={() => setIsOutputPanelOpen((current) => !current)}
          testId="output-panel"
        >
          <section className="pipeline-shell" data-testid="processing-track-grid">
            <div className="pipeline-section-heading">
              <p className="eyebrow">Pipeline</p>
              <h3>Processing Path</h3>
              <p className="summary">
                Load the video, switch pipeline steps on or off, then run the selected pipeline.
              </p>
            </div>

            <section className={`pipeline-stage-panel${isUpscaleStepEnabled ? " pipeline-stage-panel-enabled" : ""}${activePipelineVisualStep === "upscale" ? " pipeline-stage-panel-current" : ""}`} data-testid="pipeline-upscale-details">
              <div className="pipeline-stage-panel-header" data-testid="pipeline-upscale-summary">
                <div className="pipeline-stage-heading-block" data-testid="upscaler-section-card">
                  <span className="catalog-chip">Upscale</span>
                  <strong>Spatial detail restore</strong>
                  <span>{selectedModel.label}</span>
                </div>
                <button type="button" role="switch" aria-checked={isUpscaleStepEnabled} className={`pipeline-switch${isUpscaleStepEnabled ? " pipeline-switch-enabled" : ""}`} data-testid="pipeline-toggle-upscale" onClick={() => setIsUpscaleStepEnabled((current) => !current)}>
                  <span className="pipeline-switch-track"><span className="pipeline-switch-thumb" /></span>
                  <span className="pipeline-switch-label">{isUpscaleStepEnabled ? "On" : "Off"}</span>
                </button>
              </div>
              {isUpscaleStepEnabled ? (
                <section className="pipeline-stage-body" data-testid="upscaler-workspace-section">
                  <label>
                    Model
                    <select
                      data-testid="model-select"
                      value={modelId}
                      onChange={(event) => {
                        const nextModelId = event.target.value as ModelId;
                        setModelId(nextModelId);
                        setPytorchRunner(recommendedPytorchRunner(nextModelId));
                      }}
                    >
                      <optgroup label="Available Now">
                        {runnableModels.map((model) => (
                          <option key={model.value} value={model.value}>{model.label}</option>
                        ))}
                      </optgroup>
                      <optgroup label="Planned">
                        {plannedModels.map((model) => (
                          <option key={model.value} value={model.value} disabled>{model.label} (not implemented)</option>
                        ))}
                      </optgroup>
                    </select>
                  </label>
                  <details className="pipeline-detail-disclosure" data-testid="model-details-card">
                    <summary className="pipeline-detail-summary">
                      <span className="source-detail-summary-label">Model notes</span>
                      <span className="source-detail-summary-value">{selectedModel.label} • {selectedBackend.label} • {selectedModel.nativeScale}x native</span>
                    </summary>
                    <div className="pipeline-detail-body">
                      <div className="catalog-card-header">
                        <strong data-testid="selected-model-label">{selectedModel.label}</strong>
                        <span className={`catalog-chip execution-${!isSelectedModelImplemented ? "planned" : isSelectedModelLaunchable ? selectedModel.executionStatus : "setup-required"}`} data-testid="selected-model-status">
                          {!isSelectedModelImplemented ? "not implemented" : isSelectedModelLaunchable ? selectedModel.executionStatus : "setup required"}
                        </span>
                      </div>
                      <p className="summary" data-testid="selected-model-summary">{selectedModel.summary}</p>
                      {!isSelectedModelImplemented ? (
                        <p className="summary" data-testid="selected-model-availability">This model is visible in the catalog but is not implemented yet, so it cannot be selected for export.</p>
                      ) : !isSelectedModelLaunchable ? (
                        <p className="summary" data-testid="selected-model-availability">{selectedModelLaunchRequirement} Configure the external runner, refresh runtime assets by loading a source, and try again.</p>
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
                      <label>
                        Rating
                        <select data-testid="model-rating-select" value={selectedModelRating !== null ? String(selectedModelRating) : ""} onChange={(event) => void saveRating(event.target.value === "" ? null : Number(event.target.value))} disabled={isSavingRating}>
                          <option value="">No saved rating</option>
                          <option value="1">1 / 5</option>
                          <option value="2">2 / 5</option>
                          <option value="3">3 / 5</option>
                          <option value="4">4 / 5</option>
                          <option value="5">5 / 5</option>
                        </select>
                      </label>
                      <p className="summary" data-testid="rating-summary">
                        {selectedModelRating !== null ? `Saved rating: ${selectedModelRating}/5. Persisted in config/model_preferences.json.` : "No saved rating yet. Ratings persist in config/model_preferences.json."}
                      </p>
                    </div>
                  </details>
                  <label>
                    GPU Device
                    <select data-testid="gpu-select" value={selectedGpuId !== null ? String(selectedGpuId) : ""} onChange={(event) => setSelectedGpuId(event.target.value === "" ? null : Number(event.target.value))} disabled={!runtime || runtime.availableGpus.length === 0}>
                      {!runtime ? <option value="">Prepare runtime first</option> : null}
                      {runtime && runtime.availableGpus.length === 0 ? <option value="">No Vulkan GPUs detected</option> : null}
                      {runtime?.availableGpus.map((gpu) => (
                        <option key={gpu.id} value={gpu.id}>{gpu.id}: {gpu.name} ({gpu.kind})</option>
                      ))}
                    </select>
                  </label>
                  {runtime ? <p className="summary">{selectedGpu ? `Using NCNN/Vulkan GPU ${selectedGpu.id}: ${selectedGpu.name}.` : "Runtime detected no explicit Vulkan GPU selection."}</p> : null}
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
                      <input data-testid="target-width-input" type="number" min={2} step={2} value={displayedWidth} onChange={(event) => setTargetWidthInput(event.target.value)} readOnly={resolutionBasis === "height"} />
                    </label>
                    <label>
                      Target Height
                      <input data-testid="target-height-input" type="number" min={2} step={2} value={displayedHeight} onChange={(event) => setTargetHeightInput(event.target.value)} readOnly={resolutionBasis === "width"} />
                    </label>
                  </div>
                  {outputMode === "cropTo4k" ? <p className="summary">Use Edit Crop to move the framing box without blocking the source player controls. The derived dimension is computed from the selected resolution basis and stays editable through the driving axis.</p> : null}
                  <label>
                    Quality Preset
                    <select data-testid="quality-preset-select" value={qualityPreset} onChange={(event) => setQualityPreset(event.target.value as QualityPreset)}>
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
                      <span className="summary">TensorRT builds a cached engine on the first run for supported PyTorch image SR models, then reuses it on later runs.</span>
                    </label>
                  ) : null}
                  <label>
                    Tile Size
                    <input data-testid="tile-size-input" type="number" min={0} step={32} value={tileSize} onChange={(event) => setTileSize(Number(event.target.value))} />
                    <span className="summary">Use 0 for auto: the selected Quality Preset applies backend-specific tile defaults. PyTorch image SR uses 512/384/256, RVRT uses 256/192/128, and NCNN uses auto/256/128 for Max/Balanced/VRAM Safe.</span>
                  </label>
                  <label>
                    CRF
                    <input data-testid="crf-input" type="number" min={0} max={51} value={crf} onChange={(event) => setCrf(Number(event.target.value))} />
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
                </section>
              ) : null}
            </section>

            <div className="pipeline-arrow pipeline-arrow-large" aria-hidden="true">
              <span className="pipeline-arrow-shaft" />
              <span className="pipeline-arrow-head" />
            </div>

            <section className={`pipeline-stage-panel${isInterpolationStepEnabled ? " pipeline-stage-panel-enabled" : ""}${activePipelineVisualStep === "interpolate" ? " pipeline-stage-panel-current" : ""}`} data-testid="pipeline-interpolation-details">
              <div className="pipeline-stage-panel-header" data-testid="pipeline-interpolation-summary">
                <div className="pipeline-stage-heading-block" data-testid="interpolator-section-card">
                  <span className="catalog-chip">{MOTION_SECTION_NAME}</span>
                  <strong>Upsampling / interpolation</strong>
                  <span>{!isInterpolationStepEnabled ? "Off" : interpolationMode === "interpolateOnly" ? `${interpolationTargetFps} fps standalone` : `${interpolationTargetFps} fps after upscale`}</span>
                </div>
                <button type="button" role="switch" aria-checked={isInterpolationStepEnabled} className={`pipeline-switch${isInterpolationStepEnabled ? " pipeline-switch-enabled" : ""}`} data-testid="pipeline-toggle-interpolation" onClick={() => setIsInterpolationStepEnabled((current) => !current)}>
                  <span className="pipeline-switch-track"><span className="pipeline-switch-thumb" /></span>
                  <span className="pipeline-switch-label">{isInterpolationStepEnabled ? "On" : "Off"}</span>
                </button>
              </div>
              {isInterpolationStepEnabled ? (
                <section className="pipeline-stage-body pipeline-stage-body-muted" data-testid="frame-rate-workspace-section">
                  <div className="dual-field-grid">
                    <label>
                      Target Frame Rate
                      <select data-testid="frame-rate-target-select" value={String(interpolationTargetFps)} onChange={(event) => setInterpolationTargetFps(Number(event.target.value) as InterpolationTargetFps)} disabled={!interpolationEnabled}>
                        {interpolationTargetFpsOptions.map((fps) => (
                          <option key={fps} value={String(fps)}>{fps} fps</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Pipeline Step
                      <input data-testid="frame-rate-mode-readout" type="text" value={isUpscaleStepEnabled ? "Enabled after upscale" : "Enabled standalone"} readOnly />
                    </label>
                  </div>
                  <p className="summary" data-testid="interpolation-workspace-summary">
                    {interpolationMode === "interpolateOnly" ? "Interpolate-only mode is wired first so existing videos can move to a higher frame rate without requiring spatial upscaling." : "Post-upscale interpolation will synthesize frames after the spatial upscale stage so the generated motion runs at final output resolution."}
                  </p>
                </section>
              ) : null}
            </section>

            <div className="pipeline-arrow pipeline-arrow-large" aria-hidden="true">
              <span className="pipeline-arrow-shaft" />
              <span className="pipeline-arrow-head" />
            </div>

            <section className="run-results-section pipeline-run-box">
              <div className="pipeline-section-heading">
                <p className="eyebrow">Pipeline</p>
                <h3>Run Pipeline</h3>
                <p className="summary">Choose the destination file and run the enabled steps in order.</p>
              </div>
            <section className="pipeline-export-settings" data-testid="pipeline-export-settings">
              <div className="pipeline-export-settings-header">
                <strong>Export Format</strong>
                <button
                  type="button"
                  data-testid="match-input-format-button"
                  className="action-button secondary-button pipeline-inline-action"
                  onClick={matchInputFormat}
                  disabled={!source || !canMatchInputFormat}
                >
                  Match Input
                </button>
              </div>
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
              <p className="summary" data-testid="match-input-format-summary">{matchInputFormatSummary}</p>
              <details className="pipeline-detail-disclosure" data-testid="encoding-details-card">
                <summary className="pipeline-detail-summary">
                  <span className="source-detail-summary-label">Encoding Details</span>
                  <span className="source-detail-summary-value">{encodingDetailsSummary}</span>
                </summary>
                <div className="pipeline-detail-body">
                  <div className="dual-field-grid">
                    <label>
                      Quality Preset
                      <select data-testid="encoding-quality-preset-select" value={qualityPreset} onChange={(event) => setQualityPreset(event.target.value as QualityPreset)}>
                        {qualityPresets.map((preset) => (
                          <option key={preset.value} value={preset.value}>{preset.label}</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      CRF
                      <input data-testid="encoding-crf-input" type="number" min={0} max={51} value={crf} onChange={(event) => setCrf(Number(event.target.value))} />
                    </label>
                  </div>
                  <label>
                    Tile Size
                    <input data-testid="encoding-tile-size-input" type="number" min={0} step={32} value={tileSize} onChange={(event) => setTileSize(Number(event.target.value))} />
                    <span className="summary">Use 0 for auto: PyTorch defaults to 384 on balanced quality, NCNN defaults to 256.</span>
                  </label>
                </div>
              </details>
            </section>
            <section className="pipeline-export-settings" data-testid="pipeline-preview-settings">
              <div className="pipeline-export-settings-header">
                <strong>Quick Test</strong>
              </div>
              <label className="checkbox-row">
                <input data-testid="preview-mode-checkbox" type="checkbox" checked={previewMode} onChange={(event) => setPreviewMode(event.target.checked)} />
                <span>Quick Test Mode</span>
              </label>
              <label>
                Preview Duration Seconds
                <input data-testid="preview-duration-input" type="number" min={1} max={30} step={1} value={previewDurationInput} onChange={(event) => setPreviewDurationInput(event.target.value)} disabled={!previewMode} />
              </label>
              <label>
                Export Chunk Seconds
                <input data-testid="segment-duration-input" type="number" min={1} max={120} step={1} value={segmentDurationInput} onChange={(event) => setSegmentDurationInput(event.target.value)} disabled={previewMode} />
                <span className="summary">Full exports buffer this many seconds per restartable chunk. Larger chunks reduce intermediate MKV overhead but reduce restart granularity. Quick Test always runs as one segment.</span>
              </label>
            </section>
            <label>
              Output File
              <div className="path-picker-row">
                <input data-testid="output-path-input" className="path-readonly-input" type="text" value={outputPath ?? defaultOutputPath(source, container, modelId)} readOnly />
                <button data-testid="save-output-button" className="action-button secondary-button" onClick={() => void chooseOutputFile()} disabled={isRunDisabled}>
                  Save As
                </button>
              </div>
            </label>
            <button data-testid="run-upscale-button" className="action-button" onClick={() => void runPipeline()} disabled={isRunDisabled}>
              {isPipelineRunning ? "Running Pipeline..." : "Run Pipeline"}
            </button>
            {!hasEnabledPipelineStep ? (
              <p className="summary" data-testid="run-disabled-reason">
                Enable at least one pipeline step before running.
              </p>
            ) : null}
            {hasEnabledPipelineStep && isUpscaleStepEnabled && !isSelectedModelLaunchable ? (
              <p className="summary" data-testid="run-disabled-reason">
                {selectedModelLaunchRequirement}
              </p>
            ) : null}
            <details className="pipeline-detail-disclosure pipeline-runtime-disclosure">
              <summary className="pipeline-detail-summary">
                <span className="source-detail-summary-label">Runtime details</span>
                <span className="source-detail-summary-value">{runtimeFactsSummary}</span>
              </summary>
              <div className="pipeline-detail-body">
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
              </div>
            </details>
            </section>
          </section>
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
              <div className="result-detail-grid">
                <details className="source-detail-disclosure source-metadata-card" data-testid="result-output-details">
                  <summary className="source-detail-summary" title={buildResultOutputSummary(result, outputPathStats?.sizeBytes, workDirStats?.sizeBytes)}>
                    <span className="source-detail-summary-label">Output Details</span>
                    <span className="source-detail-summary-value">{buildResultOutputSummary(result, outputPathStats?.sizeBytes, workDirStats?.sizeBytes)}</span>
                  </summary>
                  <dl className="facts compact-facts source-detail-facts">
                    <div><dt>Output</dt><dd data-testid="result-output-path">{result.outputPath}</dd></div>
                    <div><dt>Output Size</dt><dd>{formatBytes(outputPathStats?.sizeBytes ?? 0)}</dd></div>
                    <div><dt>Work Dir</dt><dd>{result.workDir}</dd></div>
                    <div><dt>Scratch Size</dt><dd>{formatBytes(workDirStats?.sizeBytes ?? 0)}</dd></div>
                    <div><dt>Frames</dt><dd>{result.frameCount}</dd></div>
                    <div><dt>Source Media</dt><dd>{formatMediaSummary(result.sourceMedia ?? null)}</dd></div>
                    <div><dt>Output Media</dt><dd>{formatMediaSummary(result.outputMedia ?? null)}</dd></div>
                    <div><dt>Codec</dt><dd>{formatMediaLabel(result.codec)}</dd></div>
                    <div><dt>Container</dt><dd>{String(result.container).toUpperCase()}</dd></div>
                    <div><dt>Quality Preset</dt><dd>{formatQualityPresetLabel(result.effectiveSettings?.qualityPreset)}</dd></div>
                    <div><dt>Requested Tile Size</dt><dd>{result.effectiveSettings?.requestedTileSize && result.effectiveSettings.requestedTileSize > 0 ? result.effectiveSettings.requestedTileSize : "Auto"}</dd></div>
                    <div><dt>Effective Tile Size</dt><dd>{result.effectiveSettings?.effectiveTileSize ?? "Unknown"}</dd></div>
                    <div><dt>Precision</dt><dd>{result.effectiveSettings?.effectivePrecision ?? result.precision ?? "Unknown"}</dd></div>
                    <div><dt>Precision Source</dt><dd>{formatPrecisionSourceLabel(result.effectiveSettings?.precisionSource)}</dd></div>
                    <div><dt>Execution Path</dt><dd>{result.executionPath ?? "Unknown"}</dd></div>
                    <div><dt>Runner</dt><dd>{result.runner ?? "Unknown"}</dd></div>
                    <div><dt>Video Encoder</dt><dd>{result.videoEncoderLabel ?? result.videoEncoder ?? "Unknown"}</dd></div>
                    <div><dt>Average Throughput</dt><dd data-testid="result-average-throughput">{formatFramesPerSecond(result.averageThroughputFps)}</dd></div>
                    <div><dt>Effective Pixels Per Second</dt><dd data-testid="result-effective-pixels-per-second">{formatPixelsPerSecond(computePixelsPerSecond(result.outputMedia ?? null, result.averageThroughputFps))}</dd></div>
                    <div><dt>Peak Worker RAM</dt><dd>{formatPeakRam(result.resourcePeaks ?? null)}</dd></div>
                    <div><dt>Peak GPU Memory</dt><dd>{formatPeakGpuMemory(result.resourcePeaks ?? null)}</dd></div>
                    <div><dt>Stage Times</dt><dd data-testid="result-stage-timings">{formatStageTimingsSummary(result.stageTimings ?? null) ?? "Unavailable"}</dd></div>
                    <div><dt>Audio Sync</dt><dd>{result.hadAudio ? "Original audio remuxed" : "No source audio"}</dd></div>
                  </dl>
                </details>
                {result.interpolationDiagnostics ? (
                  <details className="diagnostics-details source-detail-disclosure source-metadata-card" data-testid="interpolation-diagnostics-details">
                    <summary className="source-detail-summary" data-testid="interpolation-diagnostics-summary" title={buildInterpolationDiagnosticsSummary(result)}>
                      <span className="source-detail-summary-label">Interpolation Details</span>
                      <span className="source-detail-summary-value">{buildInterpolationDiagnosticsSummary(result)}</span>
                    </summary>
                    <dl className="facts compact-facts diagnostics-facts source-detail-facts">
                      <div><dt>Mode</dt><dd data-testid="interpolation-diagnostics-mode">{result.interpolationDiagnostics.mode}</dd></div>
                      <div><dt>Source FPS</dt><dd data-testid="interpolation-diagnostics-source-fps">{result.interpolationDiagnostics.sourceFps.toFixed(3)}</dd></div>
                      <div><dt>Output FPS</dt><dd data-testid="interpolation-diagnostics-output-fps">{result.interpolationDiagnostics.outputFps.toFixed(3)}</dd></div>
                      <div><dt>Source Frames</dt><dd data-testid="interpolation-diagnostics-source-frames">{result.interpolationDiagnostics.sourceFrameCount}</dd></div>
                      <div><dt>Output Frames</dt><dd data-testid="interpolation-diagnostics-output-frames">{result.interpolationDiagnostics.outputFrameCount}</dd></div>
                      <div><dt>Segment Count</dt><dd data-testid="interpolation-diagnostics-segment-count">{result.interpolationDiagnostics.segmentCount}</dd></div>
                      <div><dt>Source Frame Limit</dt><dd data-testid="interpolation-diagnostics-segment-limit">{result.interpolationDiagnostics.segmentFrameLimit}</dd></div>
                      <div><dt>Boundary Overlap</dt><dd data-testid="interpolation-diagnostics-overlap">{result.interpolationDiagnostics.segmentOverlapFrames} frame{result.interpolationDiagnostics.segmentOverlapFrames === 1 ? "" : "s"}</dd></div>
                    </dl>
                  </details>
                ) : null}
                <details className="source-detail-disclosure source-metadata-card" data-testid="pipeline-log-details">
                  <summary className="source-detail-summary" title={buildWorkerLogSummary(result)}>
                    <span className="source-detail-summary-label">Worker Log</span>
                    <span className="source-detail-summary-value">{buildWorkerLogSummary(result)}</span>
                  </summary>
                  <div data-testid="pipeline-log" className="log-box source-detail-facts">
                    {result.log.map((line, index) => (
                      <div key={`${index}-${line.slice(0, 12)}`}>{line}</div>
                    ))}
                  </div>
                </details>
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

        </div>
      </section>
      </>
      ) : null}

      {isJobsOnlyView || isCleanupPanelOpen ? (
        <section className={`jobs-floating-window-shell${isJobsOnlyView ? " jobs-floating-window-shell-standalone" : ""}`} aria-label={isJobsOnlyView ? "Jobs window" : "Jobs window overlay"}>
          <article
            className={`panel jobs-floating-window${jobsWindowDragState ? " jobs-floating-window-dragging" : ""}${isJobsOnlyView ? " jobs-floating-window-standalone" : ""}`}
            data-testid="job-cleanup-panel"
            ref={jobsPanelRef}
            style={isJobsOnlyView ? undefined : {
              left: `${jobsWindowBounds.left}px`,
              top: `${jobsWindowBounds.top}px`,
              width: `${jobsWindowBounds.width}px`,
              height: `${jobsWindowBounds.height}px`,
            }}
          >
            <div className="jobs-floating-window-header" onMouseDown={beginJobsWindowDrag}>
              <div className="jobs-floating-window-title-block">
                <h2>Jobs</h2>
                <span className="expandable-panel-subtitle">{cleanupJobs.length} tracked and historical jobs</span>
              </div>
              <div className="jobs-floating-window-actions">
                {!isJobsOnlyView ? (
                  <button type="button" className="action-button secondary-button jobs-window-action" data-testid="jobs-window-reset" onClick={() => setJobsWindowBounds(defaultJobsWindowBounds())}>
                    Reset Window
                  </button>
                ) : null}
                {!isJobsOnlyView ? (
                  <button type="button" className="action-button secondary-button jobs-window-action" data-testid="jobs-window-close" onClick={() => void closeJobsWindow()}>
                    Close
                  </button>
                ) : null}
              </div>
            </div>
            <div className="jobs-floating-window-body">
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
                <strong>Pipeline Progress</strong>
                <span>{pipelineJob.progress.percent}%</span>
              </div>
              <div className="progress-shell large-progress">
                <div className="progress-bar" style={{ width: `${pipelineJob.progress.percent}%` }} />
              </div>
              <strong className="truncated-line" data-testid="progress-message" title={pipelineJob.progress.message}>{pipelineJob.progress.message}</strong>
              <span className="truncated-line progress-phase-label" title={pipelinePhaseLabel ?? ""}>{pipelinePhaseLabel}</span>
              <div className="progress-live-summary" data-testid="progress-live-summary">
                <strong className="truncated-line" data-testid="progress-current-activity" title={pipelineActivityTitle ?? ""}>{pipelineActivityTitle}</strong>
                <span className="truncated-line" data-testid="progress-current-detail" title={pipelineActivityDetail ?? ""}>{pipelineActivityDetail}</span>
                <span className="truncated-line" data-testid="progress-last-update" title={`Last update ${pipelineLastUpdateLabel}`}>Last update {pipelineLastUpdateLabel}</span>
              </div>
              {pipelineJob.progress.segmentIndex && pipelineJob.progress.segmentCount ? (
                <div className="progress-stat-grid">
                  <span className="truncated-line" data-testid="progress-segment-counter" title={`Segment: ${pipelineJob.progress.segmentIndex}/${pipelineJob.progress.segmentCount}`}>Segment: {pipelineJob.progress.segmentIndex}/{pipelineJob.progress.segmentCount}</span>
                  <span className="truncated-line" data-testid="progress-segment-frames" title={`Segment Frames: ${pipelineJob.progress.segmentProcessedFrames ?? 0}/${pipelineJob.progress.segmentTotalFrames ?? 0}`}>Segment Frames: {pipelineJob.progress.segmentProcessedFrames ?? 0}/{pipelineJob.progress.segmentTotalFrames ?? 0}</span>
                  {pipelineJob.progress.batchCount ? <span className="truncated-line" data-testid="progress-batch-counter" title={`Batch: ${pipelineJob.progress.batchIndex ?? 0}/${pipelineJob.progress.batchCount}`}>Batch: {pipelineJob.progress.batchIndex ?? 0}/{pipelineJob.progress.batchCount}</span> : null}
                </div>
              ) : null}
              <span className="progress-section-label">Job-wide stage progress</span>
              <div className="phase-progress-grid">
                {pipelinePhaseBars.map((phaseBar) => (
                  <div key={phaseBar.id} className="phase-progress-card" data-testid={`phase-progress-${phaseBar.id}`}>
                    <div className="phase-progress-header">
                      <strong>{phaseBar.label}</strong>
                      <span className="truncated-line" title={phaseBar.summary}>{phaseBar.summary}</span>
                    </div>
                    <div className="progress-shell phase-progress-shell">
                      <div className="progress-bar" style={{ width: `${phaseBar.value * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
              <div className="progress-stat-grid">
                <span className="truncated-line" data-testid="progress-total-frames" title={`Total Frames: ${pipelineJob.progress.totalFrames || "?"}`}>Total Frames: {pipelineJob.progress.totalFrames || "?"}</span>
                <span className="truncated-line" data-testid="progress-extracted-frames" title={`Extracted PNGs: ${pipelineJob.progress.extractedFrames}`}>Extracted PNGs: {pipelineJob.progress.extractedFrames}</span>
                <span className="truncated-line" data-testid="progress-upscaled-frames" title={`Upscaled PNGs: ${pipelineJob.progress.upscaledFrames}`}>Upscaled PNGs: {pipelineJob.progress.upscaledFrames}</span>
                <span className="truncated-line" data-testid="progress-interpolated-frames" title={`Interpolated PNGs: ${pipelineJob.progress.interpolatedFrames}`}>Interpolated PNGs: {pipelineJob.progress.interpolatedFrames}</span>
                <span className="truncated-line" data-testid="progress-encoded-frames" title={`Encoded Frames: ${pipelineJob.progress.encodedFrames}`}>Encoded Frames: {pipelineJob.progress.encodedFrames}</span>
                <span className="truncated-line" data-testid="progress-remuxed-frames" title={`Audio Remux Frames: ${pipelineJob.progress.remuxedFrames}`}>Audio Remux Frames: {pipelineJob.progress.remuxedFrames}</span>
              </div>
              <div className="progress-stat-grid">
                <span className="truncated-line" data-testid="progress-average-fps" title={`Average Throughput: ${formatFramesPerSecond(pipelineJob.progress.averageFramesPerSecond)}`}>Average Throughput: {formatFramesPerSecond(pipelineJob.progress.averageFramesPerSecond)}</span>
                <span className="truncated-line" data-testid="progress-rolling-fps" title={`Current Throughput: ${formatFramesPerSecond(pipelineJob.progress.rollingFramesPerSecond)}`}>Current Throughput: {formatFramesPerSecond(pipelineJob.progress.rollingFramesPerSecond)}</span>
                <span className="truncated-line" data-testid="progress-eta" title={`ETA: ${formatElapsedSeconds(pipelineJob.progress.estimatedRemainingSeconds)}`}>ETA: {formatElapsedSeconds(pipelineJob.progress.estimatedRemainingSeconds)}</span>
                <span className="truncated-line" data-testid="progress-elapsed" title={`Elapsed: ${formatElapsedSeconds(pipelineJob.progress.elapsedSeconds)}`}>Elapsed: {formatElapsedSeconds(pipelineJob.progress.elapsedSeconds)}</span>
                <span className="truncated-line" data-testid="progress-process-rss" title={`Worker RAM: ${formatBytes(pipelineJob.progress.processRssBytes ?? 0)}`}>Worker RAM: {formatBytes(pipelineJob.progress.processRssBytes ?? 0)}</span>
              </div>
              <div className="progress-stat-grid">
                <span className="truncated-line" data-testid="progress-gpu-memory" title={`GPU Memory: ${formatGpuMemory(pipelineJob.progress.gpuMemoryUsedBytes, pipelineJob.progress.gpuMemoryTotalBytes)}`}>GPU Memory: {formatGpuMemory(pipelineJob.progress.gpuMemoryUsedBytes, pipelineJob.progress.gpuMemoryTotalBytes)}</span>
                <span className="truncated-line" data-testid="workdir-size" title={`Job Scratch Size: ${formatBytes(progressScratchSizeBytes)}`}>Job Scratch Size: {formatBytes(progressScratchSizeBytes)}</span>
                <span className="truncated-line" data-testid="output-file-size" title={`Output Size: ${formatBytes(progressOutputSizeBytes)}`}>Output Size: {formatBytes(progressOutputSizeBytes)}</span>
                <span className="truncated-line" data-testid="progress-stage-timings" title={`Stage Times: ${formatStageTimings(pipelineJob.progress)}`}>Stage Times: {formatStageTimings(pipelineJob.progress)}</span>
              </div>
              <div className="progress-event-log" data-testid="progress-event-log">
                {pipelineProgressEvents.length > 0 ? [...pipelineProgressEvents].reverse().map((entry) => (
                  <div key={`${entry.key}-${entry.timestamp}`} className="progress-event-row">
                    <div className="progress-event-header">
                      <strong>{entry.title}</strong>
                      <span>{entry.percent}%</span>
                    </div>
                    <span className="truncated-line" title={entry.detail}>{entry.detail}</span>
                    <span className="progress-event-timestamp truncated-line" title={formatExactTimestamp(entry.timestamp)}>{formatExactTimestamp(entry.timestamp)}</span>
                  </div>
                )) : (
                  <div className="progress-event-row">
                    <strong>Waiting for live worker updates</strong>
                    <span className="truncated-line" title="The panel will add milestone entries as the pipeline advances.">The panel will add milestone entries as the pipeline advances.</span>
                  </div>
                )}
              </div>
            </div>
          ) : null}
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
                  <div className="cleanup-sort-hint" data-testid="cleanup-sort-hint">
                    <span className="cleanup-sort-hint-label">Sort Jobs</span>
                    <span className="summary">Click a column header to sort ascending first, then descending on the next click.</span>
                  </div>
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
                        <th scope="col">
                          <button type="button" className={`cleanup-sort-button${cleanupSort.column === "state" ? " cleanup-sort-button-active" : ""}`} data-testid="cleanup-sort-state" onClick={() => setCleanupSort((current) => toggleCleanupSort(current, "state"))}>
                            <span>State</span>
                            <span className="cleanup-sort-arrow">{cleanupSortIndicator(cleanupSort, "state")}</span>
                          </button>
                        </th>
                        <th scope="col">
                          <button type="button" className={`cleanup-sort-button${cleanupSort.column === "id" ? " cleanup-sort-button-active" : ""}`} data-testid="cleanup-sort-id" onClick={() => setCleanupSort((current) => toggleCleanupSort(current, "id"))}>
                            <span>Job ID</span>
                            <span className="cleanup-sort-arrow">{cleanupSortIndicator(cleanupSort, "id")}</span>
                          </button>
                        </th>
                        <th scope="col">
                          <button type="button" className={`cleanup-sort-button${cleanupSort.column === "scratchSize" ? " cleanup-sort-button-active" : ""}`} data-testid="cleanup-sort-scratch-size" onClick={() => setCleanupSort((current) => toggleCleanupSort(current, "scratchSize"))}>
                            <span>Temp Size</span>
                            <span className="cleanup-sort-arrow">{cleanupSortIndicator(cleanupSort, "scratchSize")}</span>
                          </button>
                        </th>
                        <th scope="col">
                          <button type="button" className={`cleanup-sort-button${cleanupSort.column === "outputSize" ? " cleanup-sort-button-active" : ""}`} data-testid="cleanup-sort-output-size" onClick={() => setCleanupSort((current) => toggleCleanupSort(current, "outputSize"))}>
                            <span>Output Size</span>
                            <span className="cleanup-sort-arrow">{cleanupSortIndicator(cleanupSort, "outputSize")}</span>
                          </button>
                        </th>
                        <th scope="col">
                          <button type="button" className={`cleanup-sort-button${cleanupSort.column === "updatedAt" ? " cleanup-sort-button-active" : ""}`} data-testid="cleanup-sort-updated" onClick={() => setCleanupSort((current) => toggleCleanupSort(current, "updatedAt"))}>
                            <span>Last Update</span>
                            <span className="cleanup-sort-arrow">{cleanupSortIndicator(cleanupSort, "updatedAt")}</span>
                          </button>
                        </th>
                        <th scope="col">
                          <button type="button" className={`cleanup-sort-button${cleanupSort.column === "input" ? " cleanup-sort-button-active" : ""}`} data-testid="cleanup-sort-input" onClick={() => setCleanupSort((current) => toggleCleanupSort(current, "input"))}>
                            <span>Input File</span>
                            <span className="cleanup-sort-arrow">{cleanupSortIndicator(cleanupSort, "input")}</span>
                          </button>
                        </th>
                        <th scope="col">
                          <button type="button" className={`cleanup-sort-button${cleanupSort.column === "output" ? " cleanup-sort-button-active" : ""}`} data-testid="cleanup-sort-output" onClick={() => setCleanupSort((current) => toggleCleanupSort(current, "output"))}>
                            <span>Output File</span>
                            <span className="cleanup-sort-arrow">{cleanupSortIndicator(cleanupSort, "output")}</span>
                          </button>
                        </th>
                        <th scope="col">Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredCleanupJobs.map((job) => {
                        const isExpanded = expandedCleanupJobIds.includes(job.id);
                        const runDetails = job.pipelineRunDetails;
                        const repeatRequest = resolveRepeatRequest(job);
                        const canRestartJob = job.jobKind === "pipeline" && (job.state === "interrupted" || job.state === "cancelled") && Boolean(repeatRequest);
                        const canLoadTemplate = Boolean(repeatRequest);
                        const isRecoverableStoppedJob = job.state === "interrupted" || job.state === "cancelled";
                        const templateTitle = repeatRequest?.exact
                          ? (isJobsOnlyView ? "Send this run's settings back to the main window so it is ready to rerun." : "Restore this run's settings into the current workspace so you can rerun or tweak them.")
                          : (isJobsOnlyView ? "Send this older job's recorded source/output settings to the main window, using current advanced defaults for fields that were not saved." : "Restore this older job's recorded source/output settings, using current advanced defaults for fields that were not saved.");
                        const detailTemplateLabel = repeatRequest?.exact
                          ? (isJobsOnlyView ? "Load Template In Main Window" : "Load Template")
                          : (isJobsOnlyView ? "Load Template In Main Window" : "Load Template From Job");
                        const restartTitle = isJobsOnlyView
                          ? "Send this stopped job back to the main window and restart it from the beginning."
                          : "Reload the recorded settings for this stopped job and start it again from the beginning.";
                        const outputMedia = runDetails?.outputMedia ?? null;
                        const averageThroughputFps = job.progress.averageFramesPerSecond ?? runDetails?.averageThroughputFps ?? null;
                        const averagePixelsPerSecond = computePixelsPerSecond(outputMedia, averageThroughputFps);
                        const currentPixelsPerSecond = computePixelsPerSecond(outputMedia, job.progress.rollingFramesPerSecond);
                        const stageTimings = formatStageTimingsSummary(runDetails?.stageTimings ?? null) ?? formatStageTimings(job.progress);
                        const exactRunMetadata = runDetails ? JSON.stringify(runDetails, null, 2) : null;
                        return (
                          <Fragment key={job.id}>
                            <tr className="cleanup-jobs-row" data-testid={`cleanup-job-${job.id}`}>
                              <td>
                                <div className="cleanup-state-cell">
                                  <span className="catalog-chip">{cleanupKindLabel(job)}</span>
                                  <strong>{job.state}</strong>
                                  <span className="cleanup-row-message">{job.label}</span>
                                </div>
                              </td>
                              <td data-testid={`cleanup-directory-${job.id}`}>
                                <div className="cleanup-id-cell">
                                  <strong className="cleanup-id-value">{job.id}</strong>
                                  <span className="cleanup-row-message">{job.scratchPath ? `Scratch: ${pathLabel(job.scratchPath, "No scratch")}` : "No scratch"}</span>
                                </div>
                              </td>
                              <td data-testid={`cleanup-scratch-size-${job.id}`}>{formatBytes(job.scratchSizeBytes)}</td>
                              <td data-testid={`cleanup-output-size-${job.id}`}>{formatBytes(job.outputSizeBytes)}</td>
                              <td data-testid={`cleanup-updated-${job.id}`}>{formatRelativeTime(job.updatedAt)}</td>
                              <td data-testid={`cleanup-input-${job.id}`} title={job.sourcePath ?? "No input path"}>{pathLabel(job.sourcePath, "No input")}</td>
                              <td data-testid={`cleanup-output-${job.id}`} title={job.outputPath ?? "No output path"}>{pathLabel(job.outputPath, "No output")}</td>
                              <td>
                                <div className="cleanup-row-actions">
                                  {canLoadTemplate ? (
                                    <button
                                      type="button"
                                      className="cleanup-expand-button cleanup-repeat-button"
                                      data-testid={`cleanup-row-repeat-${job.id}`}
                                      onClick={() => void repeatTrackedJob(job)}
                                      disabled={isBusy}
                                      title={templateTitle}
                                    >
                                      Load Template
                                    </button>
                                  ) : null}
                                  {canRestartJob ? (
                                    <button
                                      type="button"
                                      className="cleanup-expand-button cleanup-repeat-button"
                                      data-testid={`cleanup-row-restart-${job.id}`}
                                      onClick={() => void restartTrackedJob(job)}
                                      disabled={isBusy}
                                      title={restartTitle}
                                    >
                                      Restart
                                    </button>
                                  ) : null}
                                  <button type="button" className="cleanup-expand-button" data-testid={`cleanup-expand-${job.id}`} onClick={() => toggleCleanupJobExpanded(job.id)}>
                                    {isExpanded ? "Hide Details" : "Show Details"}
                                  </button>
                                </div>
                              </td>
                            </tr>
                            {isExpanded ? (
                              <tr className="cleanup-jobs-detail-row" data-testid={`cleanup-details-row-${job.id}`}>
                                <td colSpan={8}>
                                  <div className="cleanup-details-panel" data-testid={`cleanup-details-${job.id}`}>
                                    <div className="cleanup-details-grid">
                                      <span>Status Message: {job.message}</span>
                                      <span>Exact Updated: {formatExactTimestamp(job.updatedAt)}</span>
                                      <span>Job ID: {job.id}</span>
                                      <span>Phase: {job.phase}</span>
                                      <span>Recorded Frames: {job.recordedCount}</span>
                                      <span>Model: {job.modelId ?? "Unknown model"}</span>
                                      {isRecoverableStoppedJob ? <span>Recovery: This incomplete job can be restarted immediately or loaded as a template for edits first.</span> : null}
                                      <span>Codec: {job.codec ?? "Unknown codec"}</span>
                                      <span>Container: {job.container ?? "Unknown container"}</span>
                                      <span>Scratch Size: {formatBytes(job.scratchSizeBytes)}</span>
                                      <span>Output Size: {formatBytes(job.outputSizeBytes)}</span>
                                      <span>Average Throughput: {formatFramesPerSecond(averageThroughputFps)}</span>
                                      <span>Current Throughput: {formatFramesPerSecond(job.progress.rollingFramesPerSecond)}</span>
                                      <span>Effective Pixels Per Second: {formatPixelsPerSecond(averagePixelsPerSecond)}</span>
                                      <span>Current Pixels Per Second: {formatPixelsPerSecond(currentPixelsPerSecond)}</span>
                                      <span>Elapsed: {formatElapsedSeconds(job.progress.elapsedSeconds)}</span>
                                      <span>ETA: {formatElapsedSeconds(job.progress.estimatedRemainingSeconds)}</span>
                                      <span>Peak Worker RAM: {runDetails?.resourcePeaks ? formatPeakRam(runDetails.resourcePeaks) : formatBytes(job.progress.processRssBytes ?? 0)}</span>
                                      <span>Peak GPU Memory: {runDetails?.resourcePeaks ? formatPeakGpuMemory(runDetails.resourcePeaks) : formatGpuMemory(job.progress.gpuMemoryUsedBytes, job.progress.gpuMemoryTotalBytes)}</span>
                                      <span>Stage Times: {stageTimings}</span>
                                      {runDetails ? <span>Source Media: {formatMediaSummary(runDetails.sourceMedia ?? null)}</span> : null}
                                      {runDetails ? <span>Output Media: {formatMediaSummary(runDetails.outputMedia ?? null)}</span> : null}
                                      {runDetails ? <span>Execution Path: {runDetails.executionPath ?? "Unknown"}</span> : null}
                                      {runDetails ? <span>Precision: {runDetails.effectiveSettings?.effectivePrecision ?? runDetails.precision ?? "Unknown"}</span> : null}
                                      {runDetails ? <span>Encoder: {runDetails.videoEncoderLabel ?? runDetails.videoEncoder ?? "Unknown"}</span> : null}
                                      {runDetails ? <span>Runner: {runDetails.runner ?? "Unknown"}</span> : null}
                                      {runDetails ? <span>Quality Preset: {formatQualityPresetLabel(runDetails.effectiveSettings?.qualityPreset ?? runDetails.request.qualityPreset)}</span> : null}
                                      {runDetails ? <span>Requested Tile Size: {runDetails.request.tileSize > 0 ? runDetails.request.tileSize : "Auto"}</span> : null}
                                      {runDetails ? <span>Effective Tile Size: {runDetails.effectiveSettings?.effectiveTileSize ?? "Unknown"}</span> : null}
                                      {runDetails ? <span>Output Mode: {runDetails.request.outputMode}</span> : null}
                                      {runDetails ? <span>Resolution Basis: {runDetails.request.resolutionBasis}</span> : null}
                                      {runDetails ? <span>Interpolation: {runDetails.request.interpolationMode}{runDetails.request.interpolationTargetFps ? ` @ ${runDetails.request.interpolationTargetFps} fps` : ""}</span> : null}
                                      {runDetails ? <span>Preview Mode: {runDetails.request.previewMode ? "on" : "off"}</span> : null}
                                      {runDetails ? <span>Segment Duration: {runDetails.request.segmentDurationSeconds ? formatElapsedSeconds(runDetails.request.segmentDurationSeconds) : "auto"}</span> : null}
                                      {runDetails ? <span>GPU ID: {runDetails.request.gpuId ?? "auto"}</span> : null}
                                      {runDetails ? <span>PyTorch Runner: {runDetails.request.pytorchRunner}</span> : null}
                                      {runDetails ? <span>CRF: {runDetails.request.crf}</span> : null}
                                      {runDetails ? <span>Precision Source: {formatPrecisionSourceLabel(runDetails.effectiveSettings?.precisionSource)}</span> : null}
                                      {runDetails?.runtime ? <span>Runtime Paths: FFmpeg {runDetails.runtime.ffmpegPath} • Real-ESRGAN {runDetails.runtime.realesrganPath} • Models {runDetails.runtime.modelDir}</span> : null}
                                      <span>Input Path: {job.sourcePath ?? "No input path recorded"}</span>
                                      <span>Scratch Path: {job.scratchPath ?? "No scratch path recorded"}</span>
                                      <span>Output Path: {job.outputPath ?? "No output path recorded"}</span>
                                    </div>
                                    {exactRunMetadata ? (
                                      <details className="source-details-shell">
                                        <summary className="source-detail-summary">Exact Run Metadata</summary>
                                        <pre className="log-output-shell">{exactRunMetadata}</pre>
                                      </details>
                                    ) : null}
                                    <div className="job-progress-actions wrap-actions">
                                      {canLoadTemplate ? (
                                        <button
                                          type="button"
                                          className="action-button secondary-button"
                                          data-testid={`cleanup-repeat-${job.id}`}
                                          onClick={() => void repeatTrackedJob(job)}
                                          disabled={isBusy}
                                          title={templateTitle}
                                        >
                                          {detailTemplateLabel}
                                        </button>
                                      ) : null}
                                      {canRestartJob ? (
                                        <button
                                          type="button"
                                          className="action-button secondary-button"
                                          data-testid={`cleanup-restart-${job.id}`}
                                          onClick={() => void restartTrackedJob(job)}
                                          disabled={isBusy}
                                          title={restartTitle}
                                        >
                                          {isJobsOnlyView ? "Restart In Main Window" : "Restart Job"}
                                        </button>
                                      ) : null}
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
                                        <button type="button" className="action-button cleanup-stop-button" data-testid={`cleanup-stop-${job.id}`} onClick={job.onStop} disabled={isBusy} title="Stop this active job.">
                                          Stop
                                        </button>
                                      ) : null}
                                        {job.onPause ? (
                                          <button type="button" className="action-button secondary-button" data-testid={`cleanup-pause-${job.id}`} onClick={job.onPause} disabled={isBusy} title="Pause this active job.">
                                            Pause
                                          </button>
                                        ) : null}
                                        {job.onResume ? (
                                          <button type="button" className="action-button secondary-button" data-testid={`cleanup-resume-${job.id}`} onClick={job.onResume} disabled={isBusy} title="Resume this paused job.">
                                            Resume
                                          </button>
                                        ) : null}
                                      {job.onClearScratch ? (
                                        <button type="button" className="action-button secondary-button" data-testid={`cleanup-clear-scratch-${job.id}`} onClick={job.onClearScratch} disabled={isBusy || isPipelineRunning} title="Delete this job's scratch folder and intermediate working files. The job record stays available.">
                                          Clear Job Scratch
                                        </button>
                                      ) : null}
                                      {job.onDeleteOutput ? (
                                        <button type="button" className="action-button secondary-button" data-testid={`cleanup-delete-output-${job.id}`} onClick={job.onDeleteOutput} disabled={isBusy || isPipelineRunning || isSourceConversionRunning} title={job.jobKind === "sourceConversion" ? "Delete the converted source file created by the app for compatibility or preview use." : "Delete this job's exported output file while keeping the job history entry."}>
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
                    <button type="button" className="action-button secondary-button" data-testid="delete-input-file-button" onClick={() => void deleteManagedArtifact(source.path, "Input file", clearLoadedInput, ["Deletes the currently loaded managed input file.", "Clears it from the workspace immediately after deletion."])} disabled={isBusy || isPipelineRunning || isBlindComparisonRunning || isSourceConversionRunning} title="Delete the managed input file currently loaded into the workspace.">
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
                    <button type="button" className="action-button secondary-button" data-testid="delete-output-file-button" onClick={() => void deleteManagedArtifact(result.outputPath, "Output file", clearCurrentOutputSelection, ["Deletes the currently selected output file.", "Clears the output selection after the file is removed."])} disabled={isBusy || isPipelineRunning} title="Delete the currently selected output file from artifacts or the chosen export path.">
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
                <button type="button" className="action-button secondary-button" data-testid="clear-jobs-pool-button" onClick={() => void clearScratchPool(scratchSummary.jobsRoot.path, "Jobs scratch pool")} disabled={isBusy || isPipelineRunning || isSourceConversionRunning} title="Delete every managed job scratch directory under artifacts/jobs. Running jobs must be stopped first.">
                  Clear Jobs Pool
                </button>
                <button type="button" className="action-button secondary-button" data-testid="clear-converted-pool-button" onClick={() => void clearScratchPool(scratchSummary.convertedSourcesRoot.path, "Converted inputs pool")} disabled={isBusy || isPipelineRunning || isSourceConversionRunning} title="Delete converted MP4 compatibility copies stored under artifacts/runtime/converted-sources.">
                  Clear Converted Pool
                </button>
                <button type="button" className="action-button secondary-button" data-testid="clear-previews-pool-button" onClick={() => void clearScratchPool(scratchSummary.sourcePreviewsRoot.path, "Preview proxy pool")} disabled={isBusy || isPipelineRunning || isSourceConversionRunning} title="Delete generated preview proxy clips used for lightweight playback in the app.">
                  Clear Preview Pool
                </button>
              </div>
            </section>
          ) : null}
            </div>
          </article>
        </section>
      ) : null}

      {error ? <p data-testid="error-text" className="error-text">{error}</p> : null}
    </main>
  );
}
