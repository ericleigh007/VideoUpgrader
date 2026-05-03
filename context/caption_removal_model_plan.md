# Automatic Hard-Caption Removal Model Plan

## Direction

Caption removal must be fully automatic. Manual brushing, moving masks, tracked editor selections, and per-shot correction are not acceptable product workflows. Editor tools remain useful only as quality references, not as implementation models.

The target capability is:

1. Detect hard captions and subtitle-like overlays without user annotation.
2. Segment the exact caption pixels and their outlines/shadows while avoiding timestamps, scene text, reflections, and UI/bodycam overlays unless explicitly requested.
3. Remove the caption region with temporally stable video inpainting.
4. Run locally in the desktop worker first.
5. Extra-credit target: realtime or near-realtime browser inference using WebGPU/WebNN/WASM.

## Current Finding

The comparison pack now shows two separate problems:

- Fill quality: OpenCV, Big-LaMa, and ProPainter can all remove text when given an acceptable mask, but their artifacts differ.
- Mask quality: the `real-genz-red-white-bottom` sample demonstrates that a heuristic mask can confuse captions with bright reflections, timestamps, and red scene elements. Better inpainting does not fix missed or incorrect masks.

The next major gain should come from caption-specific mask prediction before deeper fill-model tuning.

## Candidate Architecture

### Stage 1: Caption Text Localization

Use a lightweight detector that proposes subtitle/caption regions automatically. Candidate approaches:

- Fine-tuned DBNet/CRAFT-style text detector biased toward subtitle overlays.
- YOLO/RT-DETR-style detector trained on caption-line boxes.
- OCR-assisted detector only for validation, not as a dependency for every frame.
- Temporal proposal smoothing so captions persist across frames without flicker.

Inputs should include full frame plus optional lower/middle region priors. The model must not assume all captions are at the bottom.

### Stage 2: Caption Pixel Segmentation

Use a caption-specific segmentation model to produce pixel masks from detected text regions. Candidate approaches:

- Small U-Net/DeepLab/SegFormer head trained on synthetic hard-caption overlays.
- SAM-style promptable segmentation distilled into an automatic caption mask model.
- Text-stroke segmentation that predicts fill, outline, and shadow pixels separately.

This stage should learn color/outline/shadow patterns, not just threshold white or saturated pixels.

### Stage 3: Temporal Mask Stabilization

Smooth masks over time using optical flow or learned temporal refinement:

- Reject one-frame false positives.
- Preserve rapidly changing karaoke captions.
- Track caption lines through cuts and camera motion.
- Avoid bleeding into fixed timestamps or unrelated scene text.

### Stage 4: Video Inpainting

Use the best available fill path behind the mask:

- Desktop baseline: ProPainter or similar video-aware inpainting.
- Fast baseline: OpenCV/LaMa for previews and low-quality mode.
- Future integrated model: caption-removal-specific inpaint model that expects thin text masks and uses neighboring frames.

## Training Data Strategy

### Synthetic Data

Generate clean videos, render captions, and keep the clean target as ground truth:

- Fonts, colors, outlines, shadows, strokes, karaoke highlights.
- Bottom, middle, top, multiline, and moving captions.
- Semi-transparent captions and YouTube/TikTok/bodycam styles.
- Distractors: timestamps, watermarks, signs, UI text, reflections, clothing graphics.

Synthetic data gives exact masks and exact clean targets for segmentation and inpainting metrics.

### Real Data

Use real clips for validation and hard negative mining:

- AirTag green/yellow mid-frame captions.
- Gen Z red/white bodycam captions with timestamp/reflection distractors.
- Additional caption styles from user-provided samples.

Real samples should be used to score residual text, false positives, temporal stability, and scene damage.

### Hard Negatives

Collect frames with text that should not be removed:

- Timestamps and bodycam overlays.
- Road signs and storefront signs.
- Clothing logos.
- Reflections and light bars.
- UI or recording indicators.

The model needs a concept of subtitle/caption overlays, not simply text removal.

## Browser Realtime Track

The browser target should be designed from the start, even if the first production model runs in Python/CUDA.

### Runtime Candidates

- ONNX Runtime Web with WebGPU execution provider.
- Transformers.js / WebGPU for supported segmentation backbones.
- TensorFlow.js WebGPU if model conversion is easier.
- WASM SIMD fallback for CPU-only browsers.

### Model Constraints

For realtime browser inference, prefer:

- 256p-540p processing resolution with masks upsampled to source frame.
- Quantized model variants: fp16 first, then int8/uint8 where quality allows.
- Separate fast mask model and optional slower fill model.
- Region-of-interest processing instead of full-frame inpainting when possible.

### Browser Pipeline Sketch

1. Decode video frame through WebCodecs.
2. Run caption detector/segmenter with WebGPU.
3. Temporally smooth mask in a small frame buffer.
4. Inpaint with one of:
   - WebGPU shader/patch fill for realtime preview.
   - Lightweight neural inpaint for higher quality.
   - Server/desktop fallback for final export quality.
5. Present with WebGL/WebGPU canvas and measure frame latency.

### Browser Success Targets

- Preview: 720p source, 30 fps playback, caption mask inference under 10-15 ms/frame on a strong desktop GPU browser.
- Quality export: slower than realtime is acceptable if the result is substantially better.
- Fallback: if neural inpaint is too slow, run realtime mask preview plus desktop worker final render.

## Evaluation Metrics

Keep the existing comparison metrics and add model-specific metrics:

- Caption mask precision/recall on synthetic ground truth.
- False-positive rate on hard negatives.
- Residual text score on real clips.
- Scene damage score on real clips.
- Temporal mask flicker.
- Inpaint temporal flicker.
- Runtime by resolution and runtime target: CUDA desktop, CPU, browser WebGPU.

## Implementation Phases

### Phase 1: Better Automatic Mask Dataset

- Extend synthetic sample generator to emit clean video, captioned video, and exact mask frames.
- Add red/white bodycam-style synthetic captions with timestamp/reflection hard negatives.
- Add metrics for mask IoU, false positives, and residual text proxies.

### Phase 2: Train or Adopt Caption Segmenter

- Start with a small segmentation model trained on synthetic overlays.
- Export to ONNX early.
- Compare against the current heuristic detector on all existing samples.

### Phase 3: Desktop Model Baseline

- Plug the learned mask into OpenCV, LaMa, and ProPainter fill paths.
- Measure whether better masks improve the Gen Z sample before changing fill models.

### Phase 4: Browser Prototype

- Build a minimal WebGPU/WebCodecs prototype that runs mask inference on video frames.
- Start with mask overlay visualization before attempting browser inpainting.
- Add realtime latency logging.

### Phase 5: Browser Inpaint Prototype

- Test shader/patch fill for preview.
- Test compact neural inpainting if WebGPU performance allows.
- Keep desktop final render as the high-quality path until browser quality is proven.

## Product Rule

Manual editor workflows are not product paths. They can be used only to establish a quality ceiling or collect reference examples. Any candidate we integrate must accept a video and produce captions-removed output with no user-drawn masks.