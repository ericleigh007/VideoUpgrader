import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import type {
  AppConfig,
  BlindComparisonSelectionInput,
  DesktopApi,
  OutputContainer,
  PipelineJobStatus,
  RealesrganJobRequest,
  RuntimeStatus,
  SourceVideoSummary
} from "../types";

declare global {
  interface Window {
    __UPSCALER_MOCK__?: Partial<DesktopApi>;
  }
}

function mockApi(): Partial<DesktopApi> | undefined {
  return window.__UPSCALER_MOCK__;
}

export const desktopApi: DesktopApi = {
  async selectVideoFile() {
    const mock = mockApi();
    if (mock?.selectVideoFile) {
      return mock.selectVideoFile();
    }

    const selected = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "mkv", "webm", "mov"] }]
    });
    return selected && !Array.isArray(selected) ? selected : null;
  },

  async selectOutputFile(defaultPath: string, container: OutputContainer) {
    const mock = mockApi();
    if (mock?.selectOutputFile) {
      return mock.selectOutputFile(defaultPath, container);
    }

    const selected = await save({
      defaultPath,
      filters: [{ name: container.toUpperCase(), extensions: [container] }]
    });
    return typeof selected === "string" ? selected : null;
  },

  async ensureRuntimeAssets() {
    const mock = mockApi();
    if (mock?.ensureRuntimeAssets) {
      return mock.ensureRuntimeAssets();
    }

    return invoke<RuntimeStatus>("ensure_runtime_assets");
  },

  async probeSourceVideo(sourcePath: string) {
    const mock = mockApi();
    if (mock?.probeSourceVideo) {
      return mock.probeSourceVideo(sourcePath);
    }

    return invoke<SourceVideoSummary>("probe_source_video", { sourcePath });
  },

  async getAppConfig() {
    const mock = mockApi();
    if (mock?.getAppConfig) {
      return mock.getAppConfig();
    }

    return invoke<AppConfig>("get_app_config");
  },

  async saveModelRating(modelId: string, rating: number | null) {
    const mock = mockApi();
    if (mock?.saveModelRating) {
      return mock.saveModelRating(modelId, rating);
    }

    return invoke<AppConfig>("save_model_rating", { modelId, rating });
  },

  async recordBlindComparisonSelection(selection: BlindComparisonSelectionInput) {
    const mock = mockApi();
    if (mock?.recordBlindComparisonSelection) {
      return mock.recordBlindComparisonSelection(selection);
    }

    return invoke<AppConfig>("record_blind_comparison_selection", { selection });
  },

  async startPipeline(request: RealesrganJobRequest) {
    const mock = mockApi();
    if (mock?.startPipeline) {
      return mock.startPipeline(request);
    }

    return invoke<string>("start_realesrgan_pipeline", { request });
  },

  async getPipelineJob(jobId: string) {
    const mock = mockApi();
    if (mock?.getPipelineJob) {
      return mock.getPipelineJob(jobId);
    }

    return invoke<PipelineJobStatus>("get_realesrgan_pipeline_job", { jobId });
  },

  async openPathInDefaultApp(path: string) {
    const mock = mockApi();
    if (mock?.openPathInDefaultApp) {
      return mock.openPathInDefaultApp(path);
    }

    return invoke<void>("open_path_in_default_app", { path });
  },

  toPreviewSrc(path: string) {
    const mock = mockApi();
    if (mock?.toPreviewSrc) {
      return mock.toPreviewSrc(path);
    }

    return convertFileSrc(path);
  }
};