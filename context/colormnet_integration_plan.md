# ColorMNet Integration Plan

## Current Conclusion

ColorMNet looks technically integrable into the current worker pipeline, but it is not a drop-in replacement for the current DeepRemaster-style "many loose reference stills" flow.

Its upstream inference path is built around a single exemplar image that is loaded as the initial color memory for the sequence. The upstream README explicitly says the reference frame and input frames should be the same size, and the CLI flag `--FirstFrameIsNotExemplar` means the reference can differ from the first frame but is still treated as the seed exemplar for propagation.

That makes ColorMNet a good fit for:

- one cleaned-up keyframe from the target shot
- one manually colorized frame from the target clip
- one still that is very close in framing, pose, and composition to the target shot

It is a weaker fit for:

- posters
- publicity stills with different staging
- loosely related set photography
- multiple unrelated references with conflicting palettes

## What The Model Wants

The upstream data flow uses:

- grayscale input frames under `input/<video>/00000.png ...`
- a single reference image under `ref/<video>/ref.png`
- a same-sequence or near-same-sequence exemplar contract

The reference image is read as a full RGB tensor and then split into luminance and chroma when `step_AnyExemplar(...)` is called. The first reference is effectively used to seed memory, and the model propagates that color information through the clip.

This matters because the model is not a general "search these stills for palette hints" engine. It is much closer to keyframe propagation with strong spatial and temporal memory.

## Product Implication

If the real user material is usually only posters and on-set ad stills, then ColorMNet alone does not solve the whole problem.

The likely product shape is:

1. Let the user choose one anchor frame from the target clip.
2. Let the user attach one or more color stills as palette evidence.
3. Create a single ColorMNet exemplar for that anchor frame.
4. Propagate from that anchor frame through the clip with ColorMNet.

Step 3 is the hard part. Without it, posters and ad stills are often too geometrically different for ColorMNet to use effectively.

## Recommended Integration Strategy

### Phase 1: Direct ColorMNet Backend

Add ColorMNet as a new research-tier runnable colorizer, but expose it honestly as an exemplar-propagation backend rather than a generic multi-reference backend.

Implementation shape:

- add `colormnet` to `config/model_catalog.json`
- keep backend as `pytorch-image-colorization`
- mark `supportsContextInput: true`
- keep `supportedContextKinds: ["referenceImages"]` only if the UI text is updated to explain that ColorMNet really wants one exemplar-like reference
- add runtime bootstrap in `python/upscaler_worker/runtime.py`
- add `python/upscaler_worker/models/colormnet.py`
- route it through `python/upscaler_worker/models/colorizers.py`

Why this shape:

- the current worker only supports colorizers with backend `pytorch-image-colorization`
- external command support exists for video SR research backends, not for colorizers
- wrapping upstream in-process will fit the current colorization pipeline with less host-side surgery than inventing a new external colorizer command path

### Phase 2: Exemplar Preparation Layer

Add a ColorMNet-specific preprocessing step that chooses exactly one reference image.

Minimum viable version:

- user selects one reference image from the source-linked context library
- app warns that best results require a same-shot or near-same-shot still
- backend resizes reference to match extracted frame size before inference
- run ColorMNet with `FirstFrameIsNotExemplar=true`

Better version:

- user selects a target anchor frame from the preview
- app offers "Create exemplar from stills"
- generate a single anchor-frame exemplar by manual paint, semi-automatic transfer, or future assisted matching
- ColorMNet then propagates from that anchor frame

## Why Poster-Only Inputs Are A Problem

Poster and publicity still workflows usually give us color semantics but not aligned geometry.

ColorMNet's strengths are:

- stable temporal propagation
- better long-range consistency than older recurrent methods
- stronger use of an exemplar frame than DeepRemaster

ColorMNet's likely failure mode for poster-only references is:

- it can infer broad palette bias from the exemplar
- but clothing regions, props, and background objects will not map reliably unless the reference is compositionally close

So if the expected production workflow is "upload three posters and let the model figure out Dorothy's dress across the reel," ColorMNet is still not the full answer.

## Required Code Changes

### Catalog And UI

Files:

- `config/model_catalog.json`
- `src/types.ts`
- `src/lib/catalog.ts`
- `src/App.tsx`

Changes:

- add `colormnet` model entry
- optionally add a ColorMNet-specific reference mode enum later if we want to distinguish "single exemplar" from generic reference images
- update UI copy so ColorMNet asks for one exemplar reference, not an arbitrary pile of stills
- surface a warning when multiple references are selected and ColorMNet is active

### Runtime Bootstrap

Files:

- `python/upscaler_worker/runtime.py`

Changes:

- add ColorMNet repo zip or release download
- add checkpoint bootstrap for `DINOv2FeatureV6_LocalAtten_s2_154000.pth`
- vendor or bootstrap required extra dependencies
- expect Windows-specific handling for the upstream DataLoader issue mentioned in the README

### Worker Model Wrapper

Files:

- `python/upscaler_worker/models/colormnet.py`
- `python/upscaler_worker/models/colorizers.py`

Changes:

- load upstream network and checkpoint
- convert extracted RGB frames to the grayscale tensor format expected by ColorMNet
- select exactly one reference image
- resize reference to frame size before the forward pass
- map progress callback per frame
- write output PNGs back into the worker's segment colorization directory

Recommended simplification:

- do not call upstream `test.py` as a subprocess from the worker
- reuse upstream network and inference core directly inside a wrapper module, the same way DeepRemaster is integrated today

### Pipeline Contract

Files:

- `python/upscaler_worker/pipeline.py`
- `python/upscaler_worker/cli.py`
- `python/upscaler_worker/models/realesrgan.py`
- `src-tauri/src/lib.rs`

Changes:

- no major contract expansion is required for a first pass because the current request already carries `referenceImagePaths`
- for a better ColorMNet product, add an explicit `colorizationReferenceMode` or `colorizationAnchorFrame` field later
- cache key should include selected reference path and any future anchor-frame selection state

## Proposed First Deliverable

The lowest-risk first slice is:

1. Integrate ColorMNet as a runnable research colorizer.
2. Support one selected reference image only.
3. Warn that the best reference is a same-shot or near-same-shot frame.
4. Validate on a clip where we can provide one manually prepared anchor still.

That will tell us whether ColorMNet is materially better than DeepRemaster before we invest in a more ambitious poster-to-exemplar workflow.

## Recommendation

Proceed with ColorMNet only if we accept this product truth:

- ColorMNet is promising as a propagation engine.
- It is not, by itself, the poster-to-film color reasoning system we ultimately want.
- If poster-only reference is the primary real-world input, we will eventually need an exemplar construction step ahead of ColorMNet.

## Suggested Next Step

Implement the Phase 1 backend and test it with a single hand-picked or hand-prepared exemplar frame from the target clip. If that works well, then decide whether to build a poster-to-exemplar preparation workflow on top.