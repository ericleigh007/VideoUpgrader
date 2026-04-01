use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
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
    progress_path: PathBuf,
    result: Option<PipelineResult>,
    error: Option<String>,
}

struct AppState {
    jobs: Arc<Mutex<HashMap<String, PipelineJobRecord>>>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri must have a parent repository directory")
        .to_path_buf()
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
    }
}

fn read_progress(progress_path: &PathBuf, fallback: &PipelineProgress) -> PipelineProgress {
    match fs::read_to_string(progress_path) {
        Ok(content) => serde_json::from_str::<PipelineProgress>(&content).unwrap_or_else(|_| fallback.clone()),
        Err(_) => fallback.clone(),
    }
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
    let progress_path = repo_root()
        .join("artifacts")
        .join("jobs")
        .join(format!("job_{job_id}_progress.json"));

    {
        let mut jobs = state.jobs.lock().map_err(|_| "Failed to lock pipeline job store".to_string())?;
        jobs.insert(
            job_id.clone(),
            PipelineJobRecord {
                state: "queued".to_string(),
                progress_path: progress_path.clone(),
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

    if request.preview_mode {
        owned_args.push("--preview-mode".to_string());
    }

    if let Some(preview_duration_seconds) = request.preview_duration_seconds {
        owned_args.push("--preview-duration-seconds".to_string());
        owned_args.push(preview_duration_seconds.to_string());
    }

    if request.fp16 {
        owned_args.push("--fp16".to_string());
    }

    if request.gpu_id.is_none() {
        owned_args.drain(8..10);
    }

    let owned_args_for_thread = owned_args;
    let job_id_for_thread = job_id.clone();
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
                        record.state = "failed".to_string();
                        record.result = None;
                        record.error = Some(error.clone());
                        let _ = fs::write(
                            &record.progress_path,
                            serde_json::to_string(&PipelineProgress {
                                phase: "failed".to_string(),
                                percent: 100,
                                message: error,
                                processed_frames: 0,
                                total_frames: 0,
                                extracted_frames: 0,
                                upscaled_frames: 0,
                                encoded_frames: 0,
                                remuxed_frames: 0,
                            })
                            .unwrap_or_else(|_| "{}".to_string()),
                        );
                    }
                }
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
fn open_path_in_default_app(path: String) -> Result<(), String> {
    open::that(path).map_err(|error| format!("Failed to open media in the default app: {error}"))
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
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            prepare_realesrgan_job,
            ensure_runtime_assets,
            probe_source_video,
            get_app_config,
            save_model_rating,
            record_blind_comparison_selection,
            start_realesrgan_pipeline,
            get_realesrgan_pipeline_job,
            open_path_in_default_app,
            scaffold_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
