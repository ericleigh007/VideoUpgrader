import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import { desktopApi } from "./lib/desktopApi";
import { getBackendDefinition, getBlindComparisonModels, getModelDefinition, getTopRatedModels, getVisibleModels } from "./lib/catalog";
import { defaultCropRect, planOutputFraming, resolveAspectRatio, type NormalizedCropRect } from "./lib/framing";
import type {
  AppConfig,
  AspectRatioPreset,
  ModelId,
  OutputContainer,
  OutputMode,
  OutputSizingOptions,
  PipelineJobStatus,
  PipelineResult,
  QualityPreset,
  RealesrganJobRequest,
  ResolutionBasis,
  RuntimeStatus,
  SourceVideoSummary,
  VideoCodec,
} from "./types";

const models = getVisibleModels();
const blindComparisonCandidates = getBlindComparisonModels();
const topRatedModels = getTopRatedModels();

const outputModes: Array<{ value: OutputMode; label: string }> = [
  { value: "preserveAspect4k", label: "Preserve Aspect In Target" },
  { value: "cropTo4k", label: "Crop To Fill Target" },
  { value: "native4x", label: "Native 4x" }
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

function normalizeOutputPath(path: string, container: OutputContainer): string {
  const suffix = `.${container}`;
  return path.toLowerCase().endsWith(suffix) ? path : `${path}${suffix}`;
}

function defaultOutputPath(source: SourceVideoSummary | null, container: OutputContainer, modelId: ModelId): string {
  const stem = source?.path.replace(/\\/g, "/").split("/").pop()?.replace(/\.[^.]+$/, "") ?? "upscaled_output";
  const modelStem = modelId.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
  return `artifacts/outputs/${stem}_${modelStem}.${container}`;
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

function resizeCropRect(startRect: NormalizedCropRect, handle: Exclude<CropHandle, "move">, deltaX: number, deltaY: number, aspectRatio: number): NormalizedCropRect {
  const leftAnchor = handle === "ne" || handle === "se" ? startRect.left : startRect.left + startRect.width;
  const topAnchor = handle === "sw" || handle === "se" ? startRect.top : startRect.top + startRect.height;
  const signedWidth = (handle === "ne" || handle === "se" ? 1 : -1) * deltaX;
  const signedHeight = (handle === "sw" || handle === "se" ? 1 : -1) * deltaY;
  const nextWidth = Math.max(0.08, startRect.width + signedWidth + (signedHeight * aspectRatio));
  const constrainedWidth = Math.min(nextWidth, 1);
  const constrainedHeight = constrainedWidth / aspectRatio;
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
  const [tileSize, setTileSize] = useState<number>(256);
  const [crf, setCrf] = useState<number>(18);
  const [previewMode, setPreviewMode] = useState<boolean>(true);
  const [previewDurationInput, setPreviewDurationInput] = useState<string>("8");
  const [source, setSource] = useState<SourceVideoSummary | null>(null);
  const [outputPath, setOutputPath] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [appConfig, setAppConfig] = useState<AppConfig | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [pipelineJob, setPipelineJob] = useState<PipelineJobStatus | null>(null);
  const [blindComparison, setBlindComparison] = useState<BlindComparisonState | null>(null);
  const [status, setStatus] = useState<string>("Idle");
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [isSavingRating, setIsSavingRating] = useState(false);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [comparisonSampleId, setComparisonSampleId] = useState<string | null>(null);
  const [comparisonZoom, setComparisonZoom] = useState<number>(3);
  const [comparisonFocusX, setComparisonFocusX] = useState<number>(50);
  const [comparisonFocusY, setComparisonFocusY] = useState<number>(50);
  const [comparisonFocusPresetId, setComparisonFocusPresetId] = useState<string>(comparisonFocusPresets[0]?.id ?? "dithering");
  const [comparisonCurrentTime, setComparisonCurrentTime] = useState<number>(0);
  const [comparisonDuration, setComparisonDuration] = useState<number>(0);
  const [comparisonPlaying, setComparisonPlaying] = useState<boolean>(false);
  const previewRef = useRef<HTMLDivElement | null>(null);
  const comparisonSourceVideoRef = useRef<HTMLVideoElement | null>(null);
  const comparisonOutputVideoRef = useRef<HTMLVideoElement | null>(null);

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
  const previewSrc = source ? desktopApi.toPreviewSrc(source.path) : null;
  const resultPreviewSrc = result ? desktopApi.toPreviewSrc(result.outputPath) : null;
  const previewDurationSeconds = parsePositiveIntegerInput(previewDurationInput);
  const displayedWidth = resolutionBasis === "height" && framing ? String(framing.canvas.width) : targetWidthInput;
  const displayedHeight = resolutionBasis === "width" && framing ? String(framing.canvas.height) : targetHeightInput;
  const cropOverlayStyle = source && framing && outputMode === "cropTo4k"
    ? {
        width: `${(framing.cropWindow.width / source.width) * 100}%`,
        height: `${(framing.cropWindow.height / source.height) * 100}%`,
        left: `${(framing.cropWindow.offsetX / source.width) * 100}%`,
        top: `${(framing.cropWindow.offsetY / source.height) * 100}%`
      }
    : undefined;
  const aspectRatioValue = source ? resolveAspectRatio({ width: source.width, height: source.height }, sizingOptions) : 16 / 9;
  const progressPercent = pipelineJob?.progress.percent ?? (result ? 100 : 0);
  const progressMessage = pipelineJob?.progress.message ?? status;
  const isPipelineRunning = pipelineJob?.state === "queued" || pipelineJob?.state === "running";
  const isBlindComparisonRunning = blindComparison?.state === "running";
  const selectedGpu = runtime?.availableGpus.find((gpu) => gpu.id === selectedGpuId) ?? null;
  const selectedModel = getModelDefinition(modelId);
  const selectedBackend = getBackendDefinition(selectedModel.backendId);
  const selectedModelRating = appConfig?.modelRatings[selectedModel.value]?.rating ?? null;
  const comparisonEntries = blindComparison?.entries.filter((entry) => Boolean(entry.status.result?.outputPath)) ?? [];
  const selectedComparisonEntry = comparisonEntries.find((entry) => entry.sampleId === comparisonSampleId) ?? comparisonEntries[0] ?? null;
  const selectedComparisonPreset = comparisonFocusPresets.find((preset) => preset.id === comparisonFocusPresetId) ?? comparisonFocusPresets[0];
  const isRunDisabled = isBusy || !source || isBlindComparisonRunning || isPipelineRunning;
  const isBlindComparisonDisabled = isBusy || !source || isBlindComparisonRunning || isPipelineRunning || blindComparisonCandidates.length < 2;

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

      setCropRect(resizeCropRect(activeDragState.startRect, activeDragState.handle, deltaX, deltaY, aspectRatioValue));
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
      setOutputPath(defaultOutputPath(summary, container, modelId));
      setResult(null);
      setPipelineJob(null);
      setActiveJobId(null);
      setBlindComparison(null);
      setComparisonSampleId(null);
      setComparisonCurrentTime(0);
      setComparisonDuration(0);
      setComparisonPlaying(false);
      setStatus("Source loaded.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStatus("Source load failed.");
    } finally {
      setIsBusy(false);
    }
  }

  async function runPipeline(): Promise<void> {
    if (!source) {
      setError("Select an mp4, mkv, webm, or mov file first.");
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

  function beginCropDrag(handle: CropHandle, event: ReactMouseEvent<HTMLElement>): void {
    if (!cropRect) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    setDragState({
      handle,
      startX: event.clientX,
      startY: event.clientY,
      startRect: cropRect
    });
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

      <section className="workspace-grid">
        <article className="panel source-panel">
          <h2>Source</h2>
          <button data-testid="select-video-button" className="action-button" onClick={() => void selectVideo()} disabled={isBusy || isPipelineRunning || isBlindComparisonRunning}>
            Select Video
          </button>
          {previewSrc ? (
            <div
              className={`preview-shell interactive-preview${outputMode === "cropTo4k" ? " crop-enabled" : ""}`}
              ref={previewRef}
              style={source ? { aspectRatio: `${source.width} / ${source.height}` } : undefined}
            >
              <video
                data-testid="source-preview"
                className="source-preview"
                controls
                preload="metadata"
                src={previewSrc}
              />
              {cropOverlayStyle ? (
                <div data-testid="crop-overlay" className="crop-overlay" style={cropOverlayStyle} onMouseDown={(event) => beginCropDrag("move", event)}>
                  <button type="button" data-testid="crop-handle-nw" className="crop-handle handle-nw" onMouseDown={(event) => beginCropDrag("nw", event)} aria-label="Resize crop from top left" />
                  <button type="button" data-testid="crop-handle-ne" className="crop-handle handle-ne" onMouseDown={(event) => beginCropDrag("ne", event)} aria-label="Resize crop from top right" />
                  <button type="button" data-testid="crop-handle-sw" className="crop-handle handle-sw" onMouseDown={(event) => beginCropDrag("sw", event)} aria-label="Resize crop from bottom left" />
                  <button type="button" data-testid="crop-handle-se" className="crop-handle handle-se" onMouseDown={(event) => beginCropDrag("se", event)} aria-label="Resize crop from bottom right" />
                </div>
              ) : null}
            </div>
          ) : (
            <div className="preview-shell preview-placeholder">
              <span>Input preview appears here after you select a video.</span>
            </div>
          )}
          {source ? (
            <dl className="facts">
              <div><dt>Path</dt><dd>{source.path}</dd></div>
              <div><dt>Resolution</dt><dd>{source.width} x {source.height}</dd></div>
              <div><dt>Duration</dt><dd>{source.durationSeconds.toFixed(2)}s</dd></div>
              <div><dt>Frame Rate</dt><dd>{source.frameRate.toFixed(3)} fps</dd></div>
              <div><dt>Audio</dt><dd>{source.hasAudio ? "Present" : "Missing"}</dd></div>
              <div><dt>Container</dt><dd>{source.container}</dd></div>
            </dl>
          ) : (
            <p className="summary">Select a local MP4, MKV, WEBM, or MOV file to probe and run.</p>
          )}
        </article>

        <article className="panel">
          <h2>Job Settings</h2>
          <label>
            Model
            <select data-testid="model-select" value={modelId} onChange={(event) => setModelId(event.target.value as ModelId)}>
              {models.map((model) => (
                <option key={model.value} value={model.value}>{model.label}</option>
              ))}
            </select>
          </label>
          <section className="catalog-card" data-testid="model-details-card">
            <div className="catalog-card-header">
              <strong data-testid="selected-model-label">{selectedModel.label}</strong>
              <span className="catalog-chip">{selectedBackend.label}</span>
            </div>
            <p className="summary" data-testid="selected-model-summary">{selectedModel.summary}</p>
            <dl className="facts compact-facts">
              <div><dt>Backend</dt><dd>{selectedBackend.label}</dd></div>
              <div><dt>Loader</dt><dd>{selectedModel.loader}</dd></div>
              <div><dt>Runtime Model</dt><dd>{selectedModel.runtimeModelName}</dd></div>
              <div><dt>Native Scale</dt><dd>{selectedModel.nativeScale}x</dd></div>
              <div><dt>Suitability</dt><dd>{selectedModel.mediaSuitability.join(", ")}</dd></div>
              <div><dt>GPU Routing</dt><dd>{selectedModel.specialHandling.supportsGpuId ? "Explicit GPU id supported" : "Automatic device selection only"}</dd></div>
            </dl>
          </section>
          <section className="catalog-card" data-testid="target-model-set-card">
            <div className="catalog-card-header">
              <strong>Top 6 Target Models</strong>
              <span className="catalog-chip">Ranked for 4K video work</span>
            </div>
            <div className="target-model-grid">
              {topRatedModels.map((model) => {
                const backend = getBackendDefinition(model.backendId);
                return (
                  <article key={model.value} className="target-model-card" data-testid={`target-model-${model.value}`}>
                    <div className="target-model-header">
                      <strong>#{model.qualityRank} {model.label}</strong>
                      <span className={`execution-badge execution-${model.executionStatus}`}>{model.executionStatus}</span>
                    </div>
                    <span className="target-model-backend">{backend.label}{model.videoNative ? " • video-native" : " • frame-by-frame"}</span>
                    <p className="summary target-model-summary">{model.summary}</p>
                  </article>
                );
              })}
            </div>
            <p className="summary">
              Today&apos;s side-by-side proof path uses the {blindComparisonCandidates.length} runnable bundled models while the remaining ranked models stay cataloged for the next backend pass.
            </p>
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
            Target Aspect Ratio
            <select data-testid="aspect-ratio-select" value={aspectRatioPreset} onChange={(event) => setAspectRatioPreset(event.target.value as AspectRatioPreset)}>
              {aspectRatioPresets.map((preset) => (
                <option key={preset.value} value={preset.value}>{preset.label}</option>
              ))}
            </select>
          </label>
          {aspectRatioPreset === "custom" ? (
            <div className="dual-field-grid">
              <label>
                Custom Aspect Width
                <input data-testid="custom-aspect-width-input" type="number" min={1} step={1} value={customAspectWidthInput} onChange={(event) => setCustomAspectWidthInput(event.target.value)} />
              </label>
              <label>
                Custom Aspect Height
                <input data-testid="custom-aspect-height-input" type="number" min={1} step={1} value={customAspectHeightInput} onChange={(event) => setCustomAspectHeightInput(event.target.value)} />
              </label>
            </div>
          ) : null}
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
                min={1}
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
                min={1}
                step={2}
                value={displayedHeight}
                onChange={(event) => setTargetHeightInput(event.target.value)}
                readOnly={resolutionBasis === "width"}
              />
            </label>
          </div>
          {outputMode === "cropTo4k" ? <p className="summary">Drag the crop box in the source preview. Use the corner handles to resize while keeping the output aspect ratio locked.</p> : null}
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
          </label>
          <label>
            CRF
            <input data-testid="crf-input" type="number" min={0} max={51} value={crf} onChange={(event) => setCrf(Number(event.target.value))} />
          </label>
          <label>
            Output File
            <div className="path-picker-row">
              <input data-testid="output-path-input" type="text" value={outputPath ?? defaultOutputPath(source, container, modelId)} readOnly />
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
          <section className="blind-comparison-panel" data-testid="blind-comparison-panel">
            <div className="catalog-card-header">
              <strong>Blind Comparison</strong>
              <span className="catalog-chip">{blindComparisonCandidates.length} runnable models</span>
            </div>
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
                  <button type="button" className="action-button secondary-button" data-testid="open-sample-external-button" onClick={() => void openMediaInDefaultApp(selectedComparisonEntry.status.result?.outputPath ?? "")}>
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
                    <input data-testid="comparison-focus-x-slider" type="range" min={0} max={100} step={1} value={comparisonFocusX} onChange={(event) => setComparisonFocusPresetId("manual") || setComparisonFocusX(Number(event.target.value))} />
                  </label>
                  <label>
                    Vertical Focus
                    <input data-testid="comparison-focus-y-slider" type="range" min={0} max={100} step={1} value={comparisonFocusY} onChange={(event) => setComparisonFocusPresetId("manual") || setComparisonFocusY(Number(event.target.value))} />
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
                        src={desktopApi.toPreviewSrc(source.path)}
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
                    <button type="button" className="inspection-viewport" data-testid="comparison-sample-viewport" onClick={() => void openMediaInDefaultApp(selectedComparisonEntry.status.result?.outputPath ?? "")}>
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
          <button data-testid="run-upscale-button" className="action-button" onClick={() => void runPipeline()} disabled={isRunDisabled}>
            {isPipelineRunning ? "Upscaling..." : "Run Upscale"}
          </button>
        </article>

        <article className="panel">
          <h2>Runtime And Output</h2>
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
          {pipelineJob ? (
            <div className="job-progress-panel" data-testid="job-progress-panel">
              <div className="progress-shell large-progress">
                <div className="progress-bar" style={{ width: `${pipelineJob.progress.percent}%` }} />
              </div>
              <strong data-testid="progress-message">{pipelineJob.progress.message}</strong>
              <span>{pipelineJob.progress.phase}</span>
              <div className="progress-stat-grid">
                <span data-testid="progress-total-frames">Total Frames: {pipelineJob.progress.totalFrames || "?"}</span>
                <span data-testid="progress-extracted-frames">Extracted PNGs: {pipelineJob.progress.extractedFrames}</span>
                <span data-testid="progress-upscaled-frames">Upscaled PNGs: {pipelineJob.progress.upscaledFrames}</span>
                <span data-testid="progress-encoded-frames">Encoded Frames: {pipelineJob.progress.encodedFrames}</span>
                <span data-testid="progress-remuxed-frames">Audio Remux Frames: {pipelineJob.progress.remuxedFrames}</span>
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
                <div><dt>Work Dir</dt><dd>{result.workDir}</dd></div>
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
          {error ? <p data-testid="error-text" className="error-text">{error}</p> : null}
        </article>
      </section>
    </main>
  );
}
