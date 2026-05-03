# Hard-Caption Removal Comparison

This document tracks how to compare the local caption-removal spike against automatic caption-removal and inpainting approaches.

Manual editor workflows are no longer product candidates. They can be kept as quality references, but any viable implementation must remove hard captions automatically from video without user-drawn masks, tracked selections, or per-shot manual correction. The implementation roadmap lives in `context/caption_removal_model_plan.md`.

## Candidate Tools

| Tool | Type | Local/cloud | Video-aware | Automation | Expected use |
| --- | --- | --- | --- | --- | --- |
| Current OpenCV spike | Heuristic mask + classical inpaint | Local | Limited | Yes | Fast baseline and mask diagnostics |
| Adobe After Effects Content-Aware Fill | Professional video editor | Local app | Yes | Manual | Quality reference only; not a product path |
| DaVinci Resolve Studio Object Removal / Fusion paint | Professional video editor | Local app | Yes | Manual | Quality reference only; not a product path |
| Runway video inpainting/object removal | Web video editor | Cloud | Yes | Manual/API depends on plan | Quality reference only unless API can run fully automatic |
| Photoshop Generative Fill | Image editor | Cloud-assisted | No | Manual/actions | Still-frame quality ceiling only |
| IOPaint / LaMa / Big-LaMa | Image inpainting model | Local | No | Yes | Practical local model baseline per frame |
| ProPainter | Video inpainting model | Local GPU | Yes | Yes | Best open-source direction for temporal quality |
| E2FGVI / STTN-style video inpainting | Video inpainting model | Local GPU | Yes | Yes | Useful older open-source baselines |
| Caption-specific detector/segmenter | Learned mask model | Local/browser-capable | Mask only | Yes | Needed next; solves mask quality before fill tuning |

## Test Samples

Store comparison clips in `artifacts/caption-removal-comparison/sources/`.

| ID | Source | Subtitle type | Why it matters |
| --- | --- | --- | --- |
| `real-airtag-green-midframe` | AirTag WebM excerpt | Green/yellow outlined captions across mid-frame | Real YouTube-style hard subtitles with scene text false positives |
| `real-genz-red-white-bottom` | Gen Z Meltdowns WebM excerpt | Red/white bottom captions over bodycam footage | Tests mixed-color caption text, bright car reflections, and timestamp false positives |
| `synthetic-white-bottom` | Generated | White bottom subtitles with dark outline | Common hard subtitle case |
| `synthetic-multiline-bottom` | Generated | Two-line white subtitles over moving background | Tests line grouping and larger masks |
| `synthetic-karaoke-moving` | Generated | Colored moving/karaoke text | Tests colored subtitle detection and temporal stability |

## Metrics

Capture both quantitative and review metrics for each tool/sample pair:

- `runtimeSeconds`: total processing time for the sample.
- `manualMinutes`: approximate human interaction time.
- `residualTextScore`: 0 means text fully readable; 5 means no readable text remains.
- `fillQualityScore`: 0 means severe smear/ghosting; 5 means background looks natural.
- `temporalStabilityScore`: 0 means distracting flicker; 5 means stable over playback.
- `sceneDamageScore`: 0 means important scene content damaged; 5 means no visible collateral damage.
- `notes`: short reviewer notes about artifacts, false positives, or workflow friction.

## Review Protocol

1. Export each editor/model result as MP4 with the same sample duration and resolution when possible.
2. Save a representative still from the middle of the clip for quick inspection.
3. Save the mask or overlay when the tool exposes it.
4. Compare against the local baseline side-by-side, not only as still frames.
5. Reject any method that removes text in stills but flickers obviously during playback.

## Current Local Baseline Finding

The OpenCV baseline can now make the AirTag title unreadable using word-cluster masks plus companion colored-pixel absorption. The remaining limitation is fill quality: classical inpaint leaves a smeared patch on textured backgrounds. That makes LaMa/Big-LaMa and ProPainter the most important next baselines.

This finding has been superseded by the Gen Z bodycam sample for the next work item: fill quality matters, but mask quality is the current blocker for mixed-color captions with hard negatives.

## Local Model Comparison Findings

The first local model pass used the same detected masks for all tools so the comparison isolates the fill engine:

- `local-opencv`: current Telea inpaint baseline.
- `simple-lama`: Big-LaMa frame inpainting via an isolated Python environment under `artifacts/runtime/lama-env`.
- `propainter`: ProPainter video inpainting cloned under `artifacts/runtime/propainter/ProPainter`.

Outputs are under `artifacts/caption-removal-comparison/results/`, with combined review images under `artifacts/caption-removal-comparison/frames/model-comparison/` and quantitative synthetic metrics in `artifacts/caption-removal-comparison/results/model-comparison/metrics.json`.

Initial result: the model baselines remove the readable text, but they are not an automatic win on the simple synthetic clips. OpenCV has the lowest masked-region error against the known synthetic clean backgrounds, mostly because the synthetic backgrounds are smooth and easy for local blur/fill to match. ProPainter is slower on the 8-second 720p samples, but it remains the most relevant next candidate for real-world footage where temporal consistency and moving texture matter more than the synthetic still-frame error.

The `real-genz-red-white-bottom` sample adds a separate failure mode: mask quality. It has red and white caption text over bodycam footage, plus bright vehicle reflections and a timestamp. The current detector masks some caption text but also catches reflections and leaves red caption fragments, so LaMa and ProPainter inherit the same problem. This sample should drive the next detector/mask work before deeper fill-engine tuning.

## Prompted Editor/Model Note

Prompted tools with instructions like `remove burned-in captions` are worth testing only if they can be run automatically against a whole clip. Prompt-only behavior may be useful as a quality reference, but it does not meet the product constraint if a user must brush masks, move selections, or adjust frames. The preferred test is therefore automatic caption-mask prediction plus fill model, not editor-guided removal.