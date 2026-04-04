use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RealesrganJobRequest {
    source_path: String,
    model_id: String,
    output_mode: String,
    quality_preset: String,
    pytorch_runner: String,
    gpu_id: Option<u32>,
    aspect_ratio_preset: String,
    custom_aspect_width: Option<u32>,
    custom_aspect_height: Option<u32>,
    resolution_basis: String,
    target_width: Option<u32>,
    target_height: Option<u32>,
    crop_left: Option<f64>,
    crop_top: Option<f64>,
    crop_width: Option<f64>,
    crop_height: Option<f64>,
    preview_mode: bool,
    preview_duration_seconds: Option<f64>,
    segment_duration_seconds: Option<f64>,
    output_path: String,
    codec: String,
    container: String,
    tile_size: u32,
    fp16: bool,
    crf: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuDevice {
    id: u32,
    name: String,
    kind: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RealesrganJobPlan {
    model: String,
    cache_key: String,
    command: Vec<String>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeStatus {
    ffmpeg_path: String,
    realesrgan_path: String,
    model_dir: String,
    available_gpus: Vec<GpuDevice>,
    default_gpu_id: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelRating {
    rating: u8,
    updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlindComparisonRecord {
    source_path: String,
    preview_duration_seconds: u32,
    winner_model_id: String,
    candidate_model_ids: Vec<String>,
    created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct AppConfig {
    model_ratings: HashMap<String, ModelRating>,
    blind_comparisons: Vec<BlindComparisonRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlindComparisonSelectionInput {
    source_path: String,
    preview_duration_seconds: u32,
    winner_model_id: String,
    candidate_model_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceVideoSummary {
    path: String,
    preview_path: String,
    width: u32,
    height: u32,
    duration_seconds: f64,
    frame_rate: f64,
    has_audio: bool,
    container: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PipelineResult {
    output_path: String,
    work_dir: String,
    frame_count: usize,
    had_audio: bool,
    codec: String,
    container: String,
    runtime: RuntimeStatus,
    log: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PipelineProgress {
    phase: String,
    percent: u32,
    message: String,
    processed_frames: usize,
    total_frames: usize,
    extracted_frames: usize,
    upscaled_frames: usize,
    encoded_frames: usize,
    remuxed_frames: usize,
    #[serde(default)]
    segment_index: Option<usize>,
    #[serde(default)]
    segment_count: Option<usize>,
    #[serde(default)]
    segment_processed_frames: Option<usize>,
    #[serde(default)]
    segment_total_frames: Option<usize>,
    #[serde(default)]
    batch_index: Option<usize>,
    #[serde(default)]
    batch_count: Option<usize>,
    #[serde(default)]
    elapsed_seconds: Option<f64>,
    #[serde(default)]
    average_frames_per_second: Option<f64>,
    #[serde(default)]
    rolling_frames_per_second: Option<f64>,
    #[serde(default)]
    estimated_remaining_seconds: Option<f64>,
    #[serde(default)]
    process_rss_bytes: Option<u64>,
    #[serde(default)]
    gpu_memory_used_bytes: Option<u64>,
    #[serde(default)]
    gpu_memory_total_bytes: Option<u64>,
    #[serde(default)]
    scratch_size_bytes: Option<u64>,
    #[serde(default)]
    output_size_bytes: Option<u64>,
    #[serde(default)]
    extract_stage_seconds: Option<f64>,
    #[serde(default)]
    upscale_stage_seconds: Option<f64>,
    #[serde(default)]
    encode_stage_seconds: Option<f64>,
    #[serde(default)]
    remux_stage_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PipelineJobStatus {
    job_id: String,
    state: String,
    progress: PipelineProgress,
    result: Option<PipelineResult>,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct PipelineJobRecord {
    state: String,
    source_path: String,
    progress_path: PathBuf,
    cancel_path: PathBuf,
    result: Option<PipelineResult>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceConversionJobStatus {
    job_id: String,
    state: String,
    progress: PipelineProgress,
    result: Option<SourceVideoSummary>,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct SourceConversionJobRecord {
    state: String,
    source_path: String,
    progress_path: PathBuf,
    cancel_path: PathBuf,
    result: Option<SourceVideoSummary>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PathStats {
    path: String,
    exists: bool,
    is_directory: bool,
    size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScratchStorageSummary {
    jobs_root: PathStats,
    converted_sources_root: PathStats,
    source_previews_root: PathStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManagedJobSummary {
    job_id: String,
    job_kind: String,
    label: String,
    state: String,
    source_path: Option<String>,
    model_id: Option<String>,
    codec: Option<String>,
    container: Option<String>,
    progress: PipelineProgress,
    recorded_count: usize,
    scratch_path: Option<String>,
    scratch_stats: Option<PathStats>,
    output_path: Option<String>,
    output_stats: Option<PathStats>,
    updated_at: String,
}

struct AppState {
    jobs: Arc<Mutex<HashMap<String, PipelineJobRecord>>>,
    source_conversion_jobs: Arc<Mutex<HashMap<String, SourceConversionJobRecord>>>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri must have a parent repository directory")
        .to_path_buf()
}

fn artifacts_root() -> PathBuf {
    repo_root().join("artifacts")
}

fn jobs_root() -> PathBuf {
    artifacts_root().join("jobs")
}

fn converted_sources_root() -> PathBuf {
    artifacts_root().join("runtime").join("converted-sources")
}

fn source_previews_root() -> PathBuf {
    artifacts_root().join("runtime").join("source-previews")
}

fn managed_job_summary_path(job_id: &str) -> PathBuf {
    jobs_root().join(format!("job_{job_id}_summary.json"))
}

fn python_command() -> String {
    env::var("UPSCALER_PYTHON").unwrap_or_else(|_| "python".to_string())
}

fn pythonpath() -> String {
    let worker_path = repo_root().join("python");
    match env::var("PYTHONPATH") {
        Ok(existing) if !existing.is_empty() => format!("{};{}", worker_path.display(), existing),
        _ => worker_path.display().to_string(),
    }
}

fn app_config_path() -> PathBuf {
    repo_root().join("config").join("model_preferences.json")
}

fn timestamp_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

fn read_app_config() -> Result<AppConfig, String> {
    let path = app_config_path();
    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let content = fs::read_to_string(&path)
        .map_err(|error| format!("Failed to read app config at {}: {error}", path.display()))?;
    if content.trim().is_empty() {
        return Ok(AppConfig::default());
    }

    serde_json::from_str::<AppConfig>(&content)
        .map_err(|error| format!("Failed to parse app config at {}: {error}", path.display()))
}

fn write_app_config(config: &AppConfig) -> Result<(), String> {
    let path = app_config_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create config directory {}: {error}", parent.display()))?;
    }

    let payload = serde_json::to_string_pretty(config)
        .map_err(|error| format!("Failed to serialize app config: {error}"))?;
    fs::write(&path, payload)
        .map_err(|error| format!("Failed to write app config at {}: {error}", path.display()))
}

fn run_python_json(args: &[&str]) -> Result<Value, String> {
    let output = Command::new(python_command())
        .current_dir(repo_root())
        .env("PYTHONPATH", pythonpath())
        .args(["-m", "upscaler_worker.cli"])
        .args(args)
        .output()
        .map_err(|error| format!("Failed to start Python worker: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return Err(format!("Python worker failed. stdout: {stdout} stderr: {stderr}"));
    }

    serde_json::from_slice::<Value>(&output.stdout)
        .map_err(|error| format!("Worker returned invalid JSON: {error}"))
}

fn build_cache_key(request: &RealesrganJobRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(request.source_path.as_bytes());
    hasher.update(request.model_id.as_bytes());
    hasher.update(request.output_mode.as_bytes());
    hasher.update(request.quality_preset.as_bytes());
    hasher.update(request.pytorch_runner.as_bytes());
    hasher.update(request.gpu_id.unwrap_or_default().to_le_bytes());
    hasher.update(request.aspect_ratio_preset.as_bytes());
    hasher.update(request.custom_aspect_width.unwrap_or_default().to_le_bytes());
    hasher.update(request.custom_aspect_height.unwrap_or_default().to_le_bytes());
    hasher.update(request.resolution_basis.as_bytes());
    hasher.update(request.target_width.unwrap_or_default().to_le_bytes());
    hasher.update(request.target_height.unwrap_or_default().to_le_bytes());
    hasher.update(request.crop_left.unwrap_or_default().to_le_bytes());
    hasher.update(request.crop_top.unwrap_or_default().to_le_bytes());
    hasher.update(request.crop_width.unwrap_or_default().to_le_bytes());
    hasher.update(request.crop_height.unwrap_or_default().to_le_bytes());
    hasher.update([request.preview_mode as u8]);
    hasher.update(request.preview_duration_seconds.unwrap_or_default().to_le_bytes());
    hasher.update(request.segment_duration_seconds.unwrap_or_default().to_le_bytes());
    hasher.update(request.output_path.as_bytes());
    hasher.update(request.codec.as_bytes());
    hasher.update(request.container.as_bytes());
    hasher.update(request.tile_size.to_le_bytes());
    hasher.update([request.fp16 as u8]);
    hasher.update(request.crf.to_le_bytes());
    format!("{:x}", hasher.finalize())
}

fn default_progress(phase: &str, percent: u32, message: &str) -> PipelineProgress {
    PipelineProgress {
        phase: phase.to_string(),
        percent,
        message: message.to_string(),
        processed_frames: 0,
        total_frames: 0,
        extracted_frames: 0,
        upscaled_frames: 0,
        encoded_frames: 0,
        remuxed_frames: 0,
        segment_index: None,
        segment_count: None,
        segment_processed_frames: None,
        segment_total_frames: None,
        batch_index: None,
        batch_count: None,
        elapsed_seconds: None,
        average_frames_per_second: None,
        rolling_frames_per_second: None,
        estimated_remaining_seconds: None,
        process_rss_bytes: None,
        gpu_memory_used_bytes: None,
        gpu_memory_total_bytes: None,
        scratch_size_bytes: None,
        output_size_bytes: None,
        extract_stage_seconds: None,
        upscale_stage_seconds: None,
        encode_stage_seconds: None,
        remux_stage_seconds: None,
    }
}

fn write_cancel_signal(cancel_path: &PathBuf, message: &str) -> Result<(), String> {
    if let Some(parent) = cancel_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create cancel directory {}: {error}", parent.display()))?;
    }
    fs::write(cancel_path, message)
        .map_err(|error| format!("Failed to write cancel signal at {}: {error}", cancel_path.display()))
}

fn path_size_bytes(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }

    if path.is_file() {
        return fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0);
    }

    fs::read_dir(path)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| path_size_bytes(&entry.path()))
        .sum()
}

fn collect_path_stats(path: &Path) -> PathStats {
    PathStats {
        path: path.display().to_string(),
        exists: path.exists(),
        is_directory: path.is_dir(),
        size_bytes: path_size_bytes(path),
    }
}

fn canonicalize_for_management(requested_path: &str) -> Result<PathBuf, String> {
    if requested_path.trim().is_empty() {
        return Err("Path cannot be empty".to_string());
    }

    let candidate = PathBuf::from(requested_path);
    let resolved = if candidate.is_absolute() {
        candidate
    } else {
        repo_root().join(candidate)
    };

    let artifacts = artifacts_root()
        .canonicalize()
        .map_err(|error| format!("Failed to resolve artifacts root: {error}"))?;
    let canonical = resolved
        .canonicalize()
        .map_err(|error| format!("Failed to resolve managed path {}: {error}", resolved.display()))?;

    if !canonical.starts_with(&artifacts) {
        return Err(format!("Refusing to manage path outside app artifacts: {}", canonical.display()));
    }

    Ok(canonical)
}

fn read_progress(progress_path: &PathBuf, fallback: &PipelineProgress) -> PipelineProgress {
    match fs::read_to_string(progress_path) {
        Ok(content) => serde_json::from_str::<PipelineProgress>(&content).unwrap_or_else(|_| fallback.clone()),
        Err(_) => fallback.clone(),
    }
}

fn path_modified_timestamp(path: &Path) -> String {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(timestamp_string)
}

fn build_managed_job_summary(
    job_id: &str,
    job_kind: &str,
    label: &str,
    state: &str,
    source_path: Option<&Path>,
    model_id: Option<&str>,
    codec: Option<&str>,
    container: Option<&str>,
    progress_path: &PathBuf,
    scratch_path: Option<&Path>,
    output_path: Option<&Path>,
) -> ManagedJobSummary {
    let fallback = match state {
        "queued" => default_progress("queued", 0, "Job queued"),
        "running" => default_progress("queued", 0, "Job running"),
        "succeeded" => default_progress("completed", 100, "Job completed"),
        "cancelled" => default_progress("failed", 100, "Job cancelled"),
        "failed" => default_progress("failed", 100, "Job failed"),
        _ => default_progress("queued", 0, "Job recorded"),
    };
    let progress = read_progress(progress_path, &fallback);
    let scratch_stats = scratch_path.map(collect_path_stats);
    let output_stats = output_path.map(collect_path_stats);

    ManagedJobSummary {
        job_id: job_id.to_string(),
        job_kind: job_kind.to_string(),
        label: label.to_string(),
        state: state.to_string(),
        source_path: source_path.map(|path| path.display().to_string()),
        model_id: model_id.map(|value| value.to_string()),
        codec: codec.map(|value| value.to_string()),
        container: container.map(|value| value.to_string()),
        recorded_count: progress.total_frames,
        progress,
        scratch_path: scratch_path.map(|path| path.display().to_string()),
        scratch_stats,
        output_path: output_path.map(|path| path.display().to_string()),
        output_stats,
        updated_at: path_modified_timestamp(progress_path),
    }
}

fn write_managed_job_summary(summary: &ManagedJobSummary) -> Result<(), String> {
    fs::create_dir_all(jobs_root())
        .map_err(|error| format!("Failed to create jobs directory {}: {error}", jobs_root().display()))?;
    let payload = serde_json::to_string_pretty(summary)
        .map_err(|error| format!("Failed to serialize managed job summary: {error}"))?;
    let target = managed_job_summary_path(&summary.job_id);
    fs::write(&target, payload)
        .map_err(|error| format!("Failed to write managed job summary {}: {error}", target.display()))
}

fn read_managed_job_summaries() -> Result<Vec<ManagedJobSummary>, String> {
    let root = jobs_root();
    if !root.exists() {
        return Ok(Vec::new());
    }

    let entries = fs::read_dir(&root)
        .map_err(|error| format!("Failed to read jobs directory {}: {error}", root.display()))?;
    let mut summaries = Vec::new();
    let mut seen_job_ids = HashSet::new();
    let mut legacy_progress = Vec::new();

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };

        if file_name.ends_with("_summary.json") {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(summary) = serde_json::from_str::<ManagedJobSummary>(&content) {
                    seen_job_ids.insert(summary.job_id.clone());
                    summaries.push(summary);
                }
            }
            continue;
        }

        if file_name.starts_with("job_") && file_name.ends_with("_progress.json") {
            legacy_progress.push(path);
        }
    }

    for progress_path in legacy_progress {
        let Some(file_name) = progress_path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(job_id) = file_name
            .strip_prefix("job_")
            .and_then(|value| value.strip_suffix("_progress.json"))
        else {
            continue;
        };

        if seen_job_ids.contains(job_id) {
            continue;
        }

        let is_conversion = job_id.starts_with("conv_");
        let scratch_path = if is_conversion {
            None
        } else {
            Some(jobs_root().join(format!("job_{job_id}")))
        };
        summaries.push(build_managed_job_summary(
            job_id,
            if is_conversion { "sourceConversion" } else { "pipeline" },
            if is_conversion { "Source Conversion" } else { "Upscale Job" },
            "succeeded",
            None,
            None,
            None,
            None,
            &progress_path,
            scratch_path.as_deref(),
            None,
        ));
    }

    summaries.sort_by(|left, right| {
        right
            .updated_at
            .parse::<u64>()
            .unwrap_or(0)
            .cmp(&left.updated_at.parse::<u64>().unwrap_or(0))
    });
    Ok(summaries)
}

fn model_label(model_id: &str) -> &str {
    match model_id {
        "realesrgan-x4plus" => "Real-ESRGAN x4 Plus",
        "realesrnet-x4plus" => "Real-ESRNet x4 Plus",
        "bsrgan-x4" => "BSRGAN x4",
        "swinir-realworld-x4" => "SwinIR Real-World x4",
        "realesrgan-x4plus-anime" => "Real-ESRGAN x4 Plus Anime",
        "realesr-animevideov3-x4" => "Real-ESR AnimeVideo v3 x4",
        _ => model_id,
    }
}

#[tauri::command]
fn prepare_realesrgan_job(request: RealesrganJobRequest) -> RealesrganJobPlan {
    let mut command = vec![
        "python".to_string(),
        "-m".to_string(),
        "upscaler_worker.cli".to_string(),
        "prepare-realesrgan-job".to_string(),
        "--source".to_string(),
        request.source_path.clone(),
        "--model-id".to_string(),
        request.model_id.clone(),
        "--output-mode".to_string(),
        request.output_mode.clone(),
        "--preset".to_string(),
        request.quality_preset.clone(),
        "--pytorch-runner".to_string(),
        request.pytorch_runner.clone(),
        "--gpu-id".to_string(),
        request.gpu_id.unwrap_or_default().to_string(),
        "--aspect-ratio-preset".to_string(),
        request.aspect_ratio_preset.clone(),
        "--resolution-basis".to_string(),
        request.resolution_basis.clone(),
        "--output-path".to_string(),
        request.output_path.clone(),
        "--codec".to_string(),
        request.codec.clone(),
        "--container".to_string(),
        request.container.clone(),
        "--tile-size".to_string(),
        request.tile_size.to_string(),
        "--crf".to_string(),
        request.crf.to_string(),
    ];

    if let Some(custom_aspect_width) = request.custom_aspect_width {
        command.push("--custom-aspect-width".to_string());
        command.push(custom_aspect_width.to_string());
    }

    if let Some(custom_aspect_height) = request.custom_aspect_height {
        command.push("--custom-aspect-height".to_string());
        command.push(custom_aspect_height.to_string());
    }

    if let Some(target_width) = request.target_width {
        command.push("--target-width".to_string());
        command.push(target_width.to_string());
    }

    if let Some(target_height) = request.target_height {
        command.push("--target-height".to_string());
        command.push(target_height.to_string());
    }

    if let Some(crop_left) = request.crop_left {
        command.push("--crop-left".to_string());
        command.push(crop_left.to_string());
    }

    if let Some(crop_top) = request.crop_top {
        command.push("--crop-top".to_string());
        command.push(crop_top.to_string());
    }

    if let Some(crop_width) = request.crop_width {
        command.push("--crop-width".to_string());
        command.push(crop_width.to_string());
    }

    if let Some(crop_height) = request.crop_height {
        command.push("--crop-height".to_string());
        command.push(crop_height.to_string());
    }

    if request.preview_mode {
        command.push("--preview-mode".to_string());
    }

    if let Some(preview_duration_seconds) = request.preview_duration_seconds {
        command.push("--preview-duration-seconds".to_string());
        command.push(preview_duration_seconds.to_string());
    }

    if let Some(segment_duration_seconds) = request.segment_duration_seconds {
        command.push("--segment-duration-seconds".to_string());
        command.push(segment_duration_seconds.to_string());
    }

    if request.fp16 {
        command.push("--fp16".to_string());
    }

    if request.gpu_id.is_none() {
        command.drain(8..10);
    }

    RealesrganJobPlan {
        model: model_label(&request.model_id).to_string(),
        cache_key: build_cache_key(&request),
        command,
        notes: vec![
            "This scaffold prepares a reproducible pipeline job contract.".to_string(),
            "Actual inference depends on the configured Python environment, backend runtime, and model artifacts.".to_string(),
        ],
    }
}

#[tauri::command]
fn ensure_runtime_assets() -> Result<RuntimeStatus, String> {
    let value = run_python_json(&["ensure-runtime"])?;
    serde_json::from_value(value).map_err(|error| format!("Failed to deserialize runtime status: {error}"))
}

#[tauri::command]
fn probe_source_video(source_path: String) -> Result<SourceVideoSummary, String> {
    let value = run_python_json(&["probe-video", "--source", &source_path])?;
    serde_json::from_value(value).map_err(|error| format!("Failed to deserialize source metadata: {error}"))
}

#[tauri::command]
fn start_source_conversion_to_mp4(state: tauri::State<AppState>, source_path: String) -> Result<String, String> {
    let job_id = format!(
        "conv_{}",
        &Sha256::digest(source_path.as_bytes())
            .iter()
            .map(|byte| format!("{:02x}", byte))
            .collect::<String>()[..12]
    );
    let progress_path = jobs_root().join(format!("job_{job_id}_progress.json"));
    let cancel_path = jobs_root().join(format!("job_{job_id}_cancel.signal"));
    let _ = fs::remove_file(&cancel_path);

    {
        let mut jobs = state
            .source_conversion_jobs
            .lock()
            .map_err(|_| "Failed to lock source conversion job store".to_string())?;
        jobs.insert(
            job_id.clone(),
            SourceConversionJobRecord {
                state: "queued".to_string(),
                source_path: source_path.clone(),
                progress_path: progress_path.clone(),
                cancel_path: cancel_path.clone(),
                result: None,
                error: None,
            },
        );
    }

    let jobs = Arc::clone(&state.source_conversion_jobs);
    let job_id_for_thread = job_id.clone();
    std::thread::spawn(move || {
        if let Ok(mut job_store) = jobs.lock() {
            if let Some(record) = job_store.get_mut(&job_id_for_thread) {
                record.state = "running".to_string();
            }
        }

        let run_result = run_python_json(&[
            "convert-source-to-mp4",
            "--source",
            &source_path,
            "--progress-path",
            &progress_path.display().to_string(),
            "--cancel-path",
            &cancel_path.display().to_string(),
        ])
        .and_then(|value| {
            serde_json::from_value::<SourceVideoSummary>(value)
                .map_err(|error| format!("Failed to deserialize converted source metadata: {error}"))
        });

        if let Ok(mut job_store) = jobs.lock() {
            if let Some(record) = job_store.get_mut(&job_id_for_thread) {
                match run_result {
                    Ok(result) => {
                        record.state = "succeeded".to_string();
                        record.result = Some(result);
                        record.error = None;
                    }
                    Err(error) => {
                        let was_cancelled = record.cancel_path.exists();
                        record.state = if was_cancelled { "cancelled".to_string() } else { "failed".to_string() };
                        record.result = None;
                        record.error = Some(if was_cancelled { "Job cancelled by user".to_string() } else { error.clone() });
                        let _ = fs::write(
                            &record.progress_path,
                            serde_json::to_string(&PipelineProgress {
                                phase: "failed".to_string(),
                                percent: 100,
                                message: if was_cancelled { "Job cancelled by user".to_string() } else { error },
                                processed_frames: 0,
                                total_frames: 0,
                                extracted_frames: 0,
                                upscaled_frames: 0,
                                encoded_frames: 0,
                                remuxed_frames: 0,
                                segment_index: None,
                                segment_count: None,
                                segment_processed_frames: None,
                                segment_total_frames: None,
                                batch_index: None,
                                batch_count: None,
                                elapsed_seconds: None,
                                average_frames_per_second: None,
                                rolling_frames_per_second: None,
                                estimated_remaining_seconds: None,
                                process_rss_bytes: None,
                                gpu_memory_used_bytes: None,
                                gpu_memory_total_bytes: None,
                                scratch_size_bytes: None,
                                output_size_bytes: None,
                                extract_stage_seconds: None,
                                upscale_stage_seconds: None,
                                encode_stage_seconds: None,
                                remux_stage_seconds: None,
                            })
                            .unwrap_or_else(|_| "{}".to_string()),
                        );
                    }
                }

                let summary = build_managed_job_summary(
                    &job_id_for_thread,
                    "sourceConversion",
                    "Source Conversion",
                    &record.state,
                    Some(Path::new(&record.source_path)),
                    None,
                    None,
                    record.result.as_ref().map(|value| value.container.as_str()),
                    &record.progress_path,
                    None,
                    record.result.as_ref().map(|value| Path::new(&value.path)),
                );
                let _ = write_managed_job_summary(&summary);
            }
        }
    });

    Ok(job_id)
}

#[tauri::command]
fn get_source_conversion_job(state: tauri::State<AppState>, job_id: String) -> Result<SourceConversionJobStatus, String> {
    let jobs = state
        .source_conversion_jobs
        .lock()
        .map_err(|_| "Failed to lock source conversion job store".to_string())?;
    let record = jobs
        .get(&job_id)
        .cloned()
        .ok_or_else(|| format!("Unknown source conversion job: {job_id}"))?;
    drop(jobs);

    let fallback = match record.state.as_str() {
        "queued" => default_progress("queued", 0, "Conversion queued"),
        "running" => default_progress("encoding", 0, "Preparing source conversion"),
        "succeeded" => default_progress("completed", 100, "Source conversion completed"),
        "cancelled" => default_progress("failed", 100, "Source conversion cancelled"),
        "failed" => default_progress("failed", 100, record.error.as_deref().unwrap_or("Source conversion failed")),
        _ => default_progress("queued", 0, "Preparing source conversion"),
    };
    let progress = read_progress(&record.progress_path, &fallback);

    Ok(SourceConversionJobStatus {
        job_id,
        state: record.state,
        progress,
        result: record.result,
        error: record.error,
    })
}

#[tauri::command]
fn cancel_source_conversion_job(state: tauri::State<AppState>, job_id: String) -> Result<(), String> {
    let mut jobs = state
        .source_conversion_jobs
        .lock()
        .map_err(|_| "Failed to lock source conversion job store".to_string())?;
    let record = jobs
        .get_mut(&job_id)
        .ok_or_else(|| format!("Unknown source conversion job: {job_id}"))?;

    if matches!(record.state.as_str(), "succeeded" | "failed" | "cancelled") {
        return Ok(());
    }

    write_cancel_signal(&record.cancel_path, "cancelled")?;
    record.error = Some("Job cancellation requested".to_string());
    Ok(())
}

#[tauri::command]
fn get_app_config() -> Result<AppConfig, String> {
    read_app_config()
}

#[tauri::command]
fn save_model_rating(model_id: String, rating: Option<u8>) -> Result<AppConfig, String> {
    if let Some(value) = rating {
        if !(1..=5).contains(&value) {
            return Err("Model rating must be between 1 and 5".to_string());
        }
    }

    let mut config = read_app_config()?;
    match rating {
        Some(value) => {
            config.model_ratings.insert(
                model_id,
                ModelRating {
                    rating: value,
                    updated_at: timestamp_string(),
                },
            );
        }
        None => {
            config.model_ratings.remove(&model_id);
        }
    }
    write_app_config(&config)?;
    Ok(config)
}

#[tauri::command]
fn record_blind_comparison_selection(selection: BlindComparisonSelectionInput) -> Result<AppConfig, String> {
    if selection.candidate_model_ids.len() < 2 {
        return Err("Blind comparison requires at least two candidate models".to_string());
    }
    if !selection
        .candidate_model_ids
        .iter()
        .any(|candidate| candidate == &selection.winner_model_id)
    {
        return Err("Blind comparison winner must be one of the candidate models".to_string());
    }

    let mut config = read_app_config()?;
    config.blind_comparisons.push(BlindComparisonRecord {
        source_path: selection.source_path,
        preview_duration_seconds: selection.preview_duration_seconds,
        winner_model_id: selection.winner_model_id,
        candidate_model_ids: selection.candidate_model_ids,
        created_at: timestamp_string(),
    });
    write_app_config(&config)?;
    Ok(config)
}

#[tauri::command]
fn start_realesrgan_pipeline(state: tauri::State<AppState>, request: RealesrganJobRequest) -> Result<String, String> {
    let job_id = build_cache_key(&request)[..12].to_string();
    let progress_path = jobs_root().join(format!("job_{job_id}_progress.json"));
    let cancel_path = jobs_root().join(format!("job_{job_id}_cancel.signal"));
    let _ = fs::remove_file(&cancel_path);

    {
        let mut jobs = state.jobs.lock().map_err(|_| "Failed to lock pipeline job store".to_string())?;
        jobs.insert(
            job_id.clone(),
            PipelineJobRecord {
                state: "queued".to_string(),
                source_path: request.source_path.clone(),
                progress_path: progress_path.clone(),
                cancel_path: cancel_path.clone(),
                result: None,
                error: None,
            },
        );
    }

    let mut owned_args = vec![
        "run-realesrgan-pipeline".to_string(),
        "--source".to_string(),
        request.source_path.clone(),
        "--model-id".to_string(),
        request.model_id.clone(),
        "--output-mode".to_string(),
        request.output_mode.clone(),
        "--preset".to_string(),
        request.quality_preset.clone(),
        "--pytorch-runner".to_string(),
        request.pytorch_runner.clone(),
        "--gpu-id".to_string(),
        request.gpu_id.unwrap_or_default().to_string(),
        "--aspect-ratio-preset".to_string(),
        request.aspect_ratio_preset.clone(),
        "--resolution-basis".to_string(),
        request.resolution_basis.clone(),
        "--output-path".to_string(),
        request.output_path.clone(),
        "--codec".to_string(),
        request.codec.clone(),
        "--container".to_string(),
        request.container.clone(),
        "--tile-size".to_string(),
        request.tile_size.to_string(),
        "--crf".to_string(),
        request.crf.to_string(),
    ];

    if let Some(custom_aspect_width) = request.custom_aspect_width {
        owned_args.push("--custom-aspect-width".to_string());
        owned_args.push(custom_aspect_width.to_string());
    }

    if let Some(custom_aspect_height) = request.custom_aspect_height {
        owned_args.push("--custom-aspect-height".to_string());
        owned_args.push(custom_aspect_height.to_string());
    }

    if let Some(target_width) = request.target_width {
        owned_args.push("--target-width".to_string());
        owned_args.push(target_width.to_string());
    }

    if let Some(target_height) = request.target_height {
        owned_args.push("--target-height".to_string());
        owned_args.push(target_height.to_string());
    }

    if let Some(crop_left) = request.crop_left {
        owned_args.push("--crop-left".to_string());
        owned_args.push(crop_left.to_string());
    }

    if let Some(crop_top) = request.crop_top {
        owned_args.push("--crop-top".to_string());
        owned_args.push(crop_top.to_string());
    }

    if let Some(crop_width) = request.crop_width {
        owned_args.push("--crop-width".to_string());
        owned_args.push(crop_width.to_string());
    }

    if let Some(crop_height) = request.crop_height {
        owned_args.push("--crop-height".to_string());
        owned_args.push(crop_height.to_string());
    }

    owned_args.push("--progress-path".to_string());
    owned_args.push(progress_path.display().to_string());
    owned_args.push("--cancel-path".to_string());
    owned_args.push(cancel_path.display().to_string());

    if request.preview_mode {
        owned_args.push("--preview-mode".to_string());
    }

    if let Some(preview_duration_seconds) = request.preview_duration_seconds {
        owned_args.push("--preview-duration-seconds".to_string());
        owned_args.push(preview_duration_seconds.to_string());
    }

    if let Some(segment_duration_seconds) = request.segment_duration_seconds {
        owned_args.push("--segment-duration-seconds".to_string());
        owned_args.push(segment_duration_seconds.to_string());
    }

    if request.fp16 {
        owned_args.push("--fp16".to_string());
    }

    if request.gpu_id.is_none() {
        owned_args.drain(11..13);
    }

    let owned_args_for_thread = owned_args;
    let job_id_for_thread = job_id.clone();
    let model_id_for_summary = request.model_id.clone();
    let codec_for_summary = request.codec.clone();
    let container_for_summary = request.container.clone();
    let jobs = Arc::clone(&state.jobs);

    std::thread::spawn(move || {
        if let Ok(mut job_store) = jobs.lock() {
            if let Some(record) = job_store.get_mut(&job_id_for_thread) {
                record.state = "running".to_string();
            }
        }

        let borrowed_args = owned_args_for_thread.iter().map(String::as_str).collect::<Vec<_>>();
        let run_result = run_python_json(&borrowed_args)
            .and_then(|value| serde_json::from_value::<PipelineResult>(value).map_err(|error| format!("Failed to deserialize pipeline result: {error}")));

        if let Ok(mut job_store) = jobs.lock() {
            if let Some(record) = job_store.get_mut(&job_id_for_thread) {
                match run_result {
                    Ok(result) => {
                        record.state = "succeeded".to_string();
                        record.result = Some(result);
                        record.error = None;
                    }
                    Err(error) => {
                        let was_cancelled = record.cancel_path.exists();
                        record.state = if was_cancelled { "cancelled".to_string() } else { "failed".to_string() };
                        record.result = None;
                        record.error = Some(if was_cancelled { "Job cancelled by user".to_string() } else { error.clone() });
                        let _ = fs::write(
                            &record.progress_path,
                            serde_json::to_string(&PipelineProgress {
                                phase: "failed".to_string(),
                                percent: 100,
                                message: if was_cancelled { "Job cancelled by user".to_string() } else { error },
                                processed_frames: 0,
                                total_frames: 0,
                                extracted_frames: 0,
                                upscaled_frames: 0,
                                encoded_frames: 0,
                                remuxed_frames: 0,
                                segment_index: None,
                                segment_count: None,
                                segment_processed_frames: None,
                                segment_total_frames: None,
                                batch_index: None,
                                batch_count: None,
                                elapsed_seconds: None,
                                average_frames_per_second: None,
                                rolling_frames_per_second: None,
                                estimated_remaining_seconds: None,
                                process_rss_bytes: None,
                                gpu_memory_used_bytes: None,
                                gpu_memory_total_bytes: None,
                                scratch_size_bytes: None,
                                output_size_bytes: None,
                                extract_stage_seconds: None,
                                upscale_stage_seconds: None,
                                encode_stage_seconds: None,
                                remux_stage_seconds: None,
                            })
                            .unwrap_or_else(|_| "{}".to_string()),
                        );
                    }
                }

                let scratch_dir = jobs_root().join(format!("job_{job_id_for_thread}"));
                let summary_scratch_path = record
                    .result
                    .as_ref()
                    .map(|value| PathBuf::from(&value.work_dir))
                    .unwrap_or(scratch_dir);
                let summary = build_managed_job_summary(
                    &job_id_for_thread,
                    "pipeline",
                    if record.result.is_some() { "Upscale Export" } else { "Upscale Job" },
                    &record.state,
                    Some(Path::new(&record.source_path)),
                    Some(&model_id_for_summary),
                    Some(&codec_for_summary),
                    Some(&container_for_summary),
                    &record.progress_path,
                    Some(summary_scratch_path.as_path()),
                    record.result.as_ref().map(|value| Path::new(&value.output_path)),
                );
                let _ = write_managed_job_summary(&summary);
            }
        }
    });

    Ok(job_id)
}

#[tauri::command]
fn get_realesrgan_pipeline_job(state: tauri::State<AppState>, job_id: String) -> Result<PipelineJobStatus, String> {
    let jobs = state.jobs.lock().map_err(|_| "Failed to lock pipeline job store".to_string())?;
    let record = jobs.get(&job_id).cloned().ok_or_else(|| format!("Unknown pipeline job: {job_id}"))?;
    drop(jobs);

    let fallback = match record.state.as_str() {
        "queued" => default_progress("queued", 0, "Job queued"),
        "running" => default_progress("queued", 0, "Preparing pipeline"),
        "succeeded" => default_progress("completed", 100, "Pipeline completed"),
        "cancelled" => default_progress("failed", 100, "Pipeline cancelled"),
        "failed" => default_progress("failed", 100, record.error.as_deref().unwrap_or("Pipeline failed")),
        _ => default_progress("queued", 0, "Preparing pipeline"),
    };
    let progress = read_progress(&record.progress_path, &fallback);

    Ok(PipelineJobStatus {
        job_id,
        state: record.state,
        progress,
        result: record.result,
        error: record.error,
    })
}

#[tauri::command]
fn cancel_realesrgan_pipeline_job(state: tauri::State<AppState>, job_id: String) -> Result<(), String> {
    let mut jobs = state.jobs.lock().map_err(|_| "Failed to lock pipeline job store".to_string())?;
    let record = jobs.get_mut(&job_id).ok_or_else(|| format!("Unknown pipeline job: {job_id}"))?;

    if matches!(record.state.as_str(), "succeeded" | "failed" | "cancelled") {
        return Ok(());
    }

    write_cancel_signal(&record.cancel_path, "cancelled")?;
    record.error = Some("Job cancellation requested".to_string());
    Ok(())
}

#[tauri::command]
fn get_path_stats(path: String) -> Result<PathStats, String> {
    if path.trim().is_empty() {
        return Err("Path cannot be empty".to_string());
    }

    let candidate = PathBuf::from(&path);
    let resolved = if candidate.is_absolute() {
        candidate
    } else {
        repo_root().join(candidate)
    };

    Ok(collect_path_stats(&resolved))
}

#[tauri::command]
fn get_scratch_storage_summary() -> ScratchStorageSummary {
    ScratchStorageSummary {
        jobs_root: collect_path_stats(&jobs_root()),
        converted_sources_root: collect_path_stats(&converted_sources_root()),
        source_previews_root: collect_path_stats(&source_previews_root()),
    }
}

#[tauri::command]
fn list_managed_jobs() -> Result<Vec<ManagedJobSummary>, String> {
    read_managed_job_summaries()
}

#[tauri::command]
fn delete_managed_path(path: String) -> Result<(), String> {
    let canonical = canonicalize_for_management(&path)?;
    if canonical.is_dir() {
        fs::remove_dir_all(&canonical)
            .map_err(|error| format!("Failed to delete directory {}: {error}", canonical.display()))?;
    } else if canonical.is_file() {
        fs::remove_file(&canonical)
            .map_err(|error| format!("Failed to delete file {}: {error}", canonical.display()))?;
    }

    Ok(())
}

#[tauri::command]
fn open_path_in_default_app(path: String) -> Result<(), String> {
    if path.trim().is_empty() {
        return Err("Cannot open an empty path".to_string());
    }

    let requested_path = PathBuf::from(&path);
    let resolved_path = if requested_path.is_absolute() {
        requested_path
    } else {
        repo_root().join(requested_path)
    };

    open::that(&resolved_path).map_err(|error| {
        format!(
            "Failed to open media in the default app: {} ({})",
            error,
            resolved_path.display()
        )
    })
}

#[tauri::command]
fn read_preview_file_base64(path: String) -> Result<String, String> {
    let bytes = fs::read(&path).map_err(|error| format!("Failed to read preview file {}: {error}", path))?;
    Ok(STANDARD.encode(bytes))
}

#[tauri::command]
fn scaffold_status() -> Vec<String> {
    vec![
        "Desktop shell scaffolded".to_string(),
        "Real-ESRGAN job planning wired".to_string(),
        "Synthetic benchmark generator available".to_string(),
    ]
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            source_conversion_jobs: Arc::new(Mutex::new(HashMap::new())),
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            prepare_realesrgan_job,
            ensure_runtime_assets,
            probe_source_video,
            start_source_conversion_to_mp4,
            get_source_conversion_job,
            cancel_source_conversion_job,
            get_app_config,
            save_model_rating,
            record_blind_comparison_selection,
            start_realesrgan_pipeline,
            get_realesrgan_pipeline_job,
            cancel_realesrgan_pipeline_job,
            get_path_stats,
            get_scratch_storage_summary,
            list_managed_jobs,
            delete_managed_path,
            open_path_in_default_app,
            read_preview_file_base64,
            scaffold_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
