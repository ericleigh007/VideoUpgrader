import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import type {
  AppConfig,
  BlindComparisonSelectionInput,
  DesktopApi,
  ManagedJobSummary,
  OutputContainer,
  PathStats,
  PipelineJobStatus,
  RealesrganJobRequest,
  RuntimeStatus,
  ScratchStorageSummary,
  SourceConversionJobStatus,
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
      filters: [{ name: "Video", extensions: ["mp4", "mkv", "webm", "mov", "avi"] }]
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

  async startSourceConversionToMp4(sourcePath: string) {
    const mock = mockApi();
    if (mock?.startSourceConversionToMp4) {
      return mock.startSourceConversionToMp4(sourcePath);
    }

    return invoke<string>("start_source_conversion_to_mp4", { sourcePath });
  },

  async getSourceConversionJob(jobId: string) {
    const mock = mockApi();
    if (mock?.getSourceConversionJob) {
      return mock.getSourceConversionJob(jobId);
    }

    return invoke<SourceConversionJobStatus>("get_source_conversion_job", { jobId });
  },

  async pauseSourceConversionJob(jobId: string) {
    const mock = mockApi();
    if (mock?.pauseSourceConversionJob) {
      return mock.pauseSourceConversionJob(jobId);
    }

    return invoke<void>("pause_source_conversion_job", { jobId });
  },

  async resumeSourceConversionJob(jobId: string) {
    const mock = mockApi();
    if (mock?.resumeSourceConversionJob) {
      return mock.resumeSourceConversionJob(jobId);
    }

    return invoke<void>("resume_source_conversion_job", { jobId });
  },

  async cancelSourceConversionJob(jobId: string) {
    const mock = mockApi();
    if (mock?.cancelSourceConversionJob) {
      return mock.cancelSourceConversionJob(jobId);
    }

    return invoke<void>("cancel_source_conversion_job", { jobId });
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

  async pausePipelineJob(jobId: string) {
    const mock = mockApi();
    if (mock?.pausePipelineJob) {
      return mock.pausePipelineJob(jobId);
    }

    return invoke<void>("pause_realesrgan_pipeline_job", { jobId });
  },

  async resumePipelineJob(jobId: string) {
    const mock = mockApi();
    if (mock?.resumePipelineJob) {
      return mock.resumePipelineJob(jobId);
    }

    return invoke<void>("resume_realesrgan_pipeline_job", { jobId });
  },

  async cancelPipelineJob(jobId: string) {
    const mock = mockApi();
    if (mock?.cancelPipelineJob) {
      return mock.cancelPipelineJob(jobId);
    }

    return invoke<void>("cancel_realesrgan_pipeline_job", { jobId });
  },

  async getPathStats(path: string) {
    const mock = mockApi();
    if (mock?.getPathStats) {
      return mock.getPathStats(path);
    }

    return invoke<PathStats>("get_path_stats", { path });
  },

  async getScratchStorageSummary() {
    const mock = mockApi();
    if (mock?.getScratchStorageSummary) {
      return mock.getScratchStorageSummary();
    }

    return invoke<ScratchStorageSummary>("get_scratch_storage_summary");
  },

  async listManagedJobs() {
    const mock = mockApi();
    if (mock?.listManagedJobs) {
      return mock.listManagedJobs();
    }

    return invoke<ManagedJobSummary[]>("list_managed_jobs");
  },

  async deleteManagedPath(path: string) {
    const mock = mockApi();
    if (mock?.deleteManagedPath) {
      return mock.deleteManagedPath(path);
    }

    return invoke<void>("delete_managed_path", { path });
  },

  async openPathInDefaultApp(path: string) {
    const mock = mockApi();
    if (mock?.openPathInDefaultApp) {
      return mock.openPathInDefaultApp(path);
    }

    return invoke<void>("open_path_in_default_app", { path });
  },

  async loadPreviewUrl(path: string) {
    const mock = mockApi();
    if (mock?.loadPreviewUrl) {
      return mock.loadPreviewUrl(path);
    }
    if (mock?.toPreviewSrc) {
      return mock.toPreviewSrc(path);
    }

    if (/^(https?:|blob:|\/)/i.test(path)) {
      return this.toPreviewSrc(path);
    }

    return this.toPreviewSrc(path);
  },

  toPreviewSrc(path: string) {
    const mock = mockApi();
    if (mock?.toPreviewSrc) {
      return mock.toPreviewSrc(path);
    }

    return convertFileSrc(path.replace(/\\/g, "/"));
  }
};