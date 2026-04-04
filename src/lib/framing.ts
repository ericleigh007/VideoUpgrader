import type { OutputMode, OutputSizingOptions } from "../types";

export interface FrameSize {
  width: number;
  height: number;
}

export interface NormalizedCropRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface FramingPlan {
  canvas: FrameSize;
  scaled: FrameSize;
  crop: FrameSize;
  cropWindow: {
    width: number;
    height: number;
    offsetX: number;
    offsetY: number;
  };
  aspectRatio: number;
  mode: OutputMode;
}

const UHD_4K: FrameSize = { width: 3840, height: 2160 };
const EPSILON = 0.0001;

function roundDimension(value: number): number {
  const rounded = Math.max(2, Math.round(value));
  return rounded % 2 === 0 ? rounded : rounded + 1;
}

function sourceAspectRatio(source: FrameSize): number {
  return source.width / source.height;
}

function clampNormalized(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function resolveAspectRatio(source: FrameSize, options: OutputSizingOptions): number {
  if (options.aspectRatioPreset === "source") {
    return sourceAspectRatio(source);
  }

  if (options.aspectRatioPreset === "custom") {
    const width = options.customAspectWidth ?? 0;
    const height = options.customAspectHeight ?? 0;
    if (width > 0 && height > 0) {
      return width / height;
    }
    return sourceAspectRatio(source);
  }

  const [width, height] = options.aspectRatioPreset.split(":").map((value) => Number.parseFloat(value));
  return width > 0 && height > 0 ? width / height : sourceAspectRatio(source);
}

export function resolveOutputCanvas(source: FrameSize, mode: OutputMode, options: OutputSizingOptions): FrameSize {
  const aspectRatio = resolveAspectRatio(source, options);
  const requestedWidth = options.targetWidth ?? 0;
  const requestedHeight = options.targetHeight ?? 0;

  if (options.resolutionBasis === "exact" && requestedWidth > 0 && requestedHeight > 0) {
    return {
      width: roundDimension(requestedWidth),
      height: roundDimension(requestedHeight)
    };
  }

  if (options.resolutionBasis === "width" && requestedWidth > 0) {
    return {
      width: roundDimension(requestedWidth),
      height: roundDimension(requestedWidth / aspectRatio)
    };
  }

  if (options.resolutionBasis === "height" && requestedHeight > 0) {
    return {
      width: roundDimension(requestedHeight * aspectRatio),
      height: roundDimension(requestedHeight)
    };
  }

  if (requestedWidth > 0 && requestedHeight <= 0) {
    return {
      width: roundDimension(requestedWidth),
      height: roundDimension(requestedWidth / aspectRatio)
    };
  }

  if (requestedHeight > 0 && requestedWidth <= 0) {
    return {
      width: roundDimension(requestedHeight * aspectRatio),
      height: roundDimension(requestedHeight)
    };
  }

  if (mode === "native4x" && Math.abs(aspectRatio - sourceAspectRatio(source)) < EPSILON) {
    return {
      width: roundDimension(source.width * 2),
      height: roundDimension(source.height * 2)
    };
  }

  if (aspectRatio >= 1) {
    return {
      width: UHD_4K.width,
      height: roundDimension(UHD_4K.width / aspectRatio)
    };
  }

  return {
    width: roundDimension(UHD_4K.height * aspectRatio),
    height: UHD_4K.height
  };
}

export function defaultCropRect(source: FrameSize, options: OutputSizingOptions): NormalizedCropRect {
  const aspectRatio = resolveAspectRatio(source, options);
  const sourceRatio = sourceAspectRatio(source);

  if (Math.abs(aspectRatio - sourceRatio) < EPSILON) {
    return { left: 0, top: 0, width: 1, height: 1 };
  }

  if (aspectRatio > sourceRatio) {
    const height = sourceRatio / aspectRatio;
    const top = (1 - height) / 2;
    return { left: 0, top, width: 1, height };
  }

  const width = aspectRatio / sourceRatio;
  const left = (1 - width) / 2;
  return { left, top: 0, width, height: 1 };
}

export function resolveCropRect(source: FrameSize, options: OutputSizingOptions): NormalizedCropRect {
  const fallback = defaultCropRect(source, options);
  const left = options.cropLeft;
  const top = options.cropTop;
  const width = options.cropWidth;
  const height = options.cropHeight;

  if (left === null || top === null || width === null || height === null) {
    return fallback;
  }

  const resolvedWidth = clampNormalized(width);
  const resolvedHeight = clampNormalized(height);
  const maxLeft = Math.max(0, 1 - resolvedWidth);
  const maxTop = Math.max(0, 1 - resolvedHeight);

  return {
    left: Math.min(maxLeft, clampNormalized(left)),
    top: Math.min(maxTop, clampNormalized(top)),
    width: resolvedWidth,
    height: resolvedHeight
  };
}

export function planOutputFraming(source: FrameSize, mode: OutputMode, options: OutputSizingOptions): FramingPlan {
  const canvas = resolveOutputCanvas(source, mode, options);
  const aspectRatio = resolveAspectRatio(source, options);
  if (mode === "native4x" && Math.abs(aspectRatio - sourceAspectRatio(source)) < EPSILON) {
    return {
      canvas,
      scaled: canvas,
      crop: canvas,
      cropWindow: {
        width: source.width,
        height: source.height,
        offsetX: 0,
        offsetY: 0
      },
      aspectRatio,
      mode
    };
  }

  const scaleX = canvas.width / source.width;
  const scaleY = canvas.height / source.height;
  const scale = mode === "cropTo4k" ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY);

  const cropRect = mode === "cropTo4k"
    ? resolveCropRect(source, options)
    : { left: 0, top: 0, width: 1, height: 1 };

  const scaled = {
    width: roundDimension(source.width * (mode === "cropTo4k" ? canvas.width / (source.width * cropRect.width) : scale)),
    height: roundDimension(source.height * (mode === "cropTo4k" ? canvas.height / (source.height * cropRect.height) : scale))
  };

  const cropWindowWidth = roundDimension(source.width * cropRect.width);
  const cropWindowHeight = roundDimension(source.height * cropRect.height);

  return {
    canvas,
    scaled,
    crop: canvas,
    cropWindow: {
      width: Math.min(source.width, cropWindowWidth),
      height: Math.min(source.height, cropWindowHeight),
      offsetX: Math.round(source.width * cropRect.left),
      offsetY: Math.round(source.height * cropRect.top)
    },
    aspectRatio,
    mode
  };
}
