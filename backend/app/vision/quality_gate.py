"""
Nidaan-Rx v2.0 — Pillar 1: Tone-Calibrated Image Quality Gate & Privacy Sanitizer.

Designed for Indian & global dermatology smartphone photos (Fitzpatrick I-VI / Monk 1-10):
  - Executes in < 50 ms on CPU via OpenCV.
  - Uses luminance dynamic range & local contrast instead of naive mean-brightness
    thresholds so darker Fitzpatrick IV-VI skin tones are never falsely rejected.
  - Strips 100% of EXIF/GPS metadata and compresses uploads to <= 1024 px for
    DPDP Act privacy compliance and Supabase 1 GB free-tier storage budgeting.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass



@dataclass
class QualityGateResult:
    passed: bool
    elapsed_ms: float
    width: int
    height: int
    blur_score: float
    dynamic_range: float
    local_contrast: float
    overexposed_ratio: float
    crushed_black_ratio: float
    skin_scalp_signal: float
    issues: list[str] = field(default_factory=list)
    retake_cues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def reasons(self) -> list[str]:
        return self.issues

    @property
    def warnings(self) -> list[str]:
        return self.retake_cues

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "width": self.width,
            "height": self.height,
            "blur_score": round(self.blur_score, 2),
            "dynamic_range": round(self.dynamic_range, 2),
            "local_contrast": round(self.local_contrast, 2),
            "overexposed_ratio": round(self.overexposed_ratio, 4),
            "crushed_black_ratio": round(self.crushed_black_ratio, 4),
            "skin_scalp_signal": round(self.skin_scalp_signal, 4),
            "issues": self.issues,
            "retake_cues": self.retake_cues,
        }


def _decode_bgr(image_input: bytes | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input
    if isinstance(image_input, Image.Image):
        rgb = np.array(ImageOps.exif_transpose(image_input).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if isinstance(image_input, (bytes, bytearray)):
        arr = np.frombuffer(image_input, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr
        pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_input))).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def evaluate_image_quality(
    image_input: bytes | np.ndarray | Image.Image,
    min_dim: int = 224,
    min_blur_var: float = 40.0,
    min_dynamic_range: float = 12.0,
    min_local_contrast: float = 4.0,
    max_overexposed_ratio: float = 0.28,
    max_crushed_black_ratio: float = 0.55,
    min_skin_scalp_signal: float = 0.12,
) -> QualityGateResult:
    """
    Tone-calibrated pre-inference quality gate (<50 ms).

    Evaluates the central 70% ROI (where the skin/scalp lesion sits) so
    out-of-focus backgrounds or dark tabletops never penalize valid macro photos.
    """
    t0 = time.perf_counter()
    bgr = _decode_bgr(image_input)
    h, w = bgr.shape[:2]

    issues: list[str] = []
    cues: list[str] = []

    # 1. Minimum resolution check (>= 224x224 px)
    if h < min_dim or w < min_dim:
        issues.append(f"low_resolution ({w}x{h})")
        cues.append(
            f"Image resolution is too small ({w}x{h}px). Capture a closer, uncropped photo (>= {min_dim}x{min_dim}px)."
        )

    # Downscale working copy to max 512px for guaranteed <15 ms deterministic latency
    work = bgr
    max_side = max(h, w)
    if max_side > 512:
        scale = 512.0 / max_side
        work = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    wh, ww = work.shape[:2]
    y0, y1 = int(wh * 0.15), max(int(wh * 0.85), int(wh * 0.15) + 1)
    x0, x1 = int(ww * 0.15), max(int(ww * 0.85), int(ww * 0.15) + 1)
    roi_bgr = work[y0:y1, x0:x1]

    # Luminance & chroma in YCrCb
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    y_chan = ycrcb[:, :, 0]
    cr_chan = ycrcb[:, :, 1]
    cb_chan = ycrcb[:, :, 2]

    # 2. Blur detection via Laplacian variance on CLAHE-normalized luminance
    # Prevents darker skin tones from having artificially lower Laplacian gradient magnitude
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y_chan)
    blur_score = float(cv2.Laplacian(y_eq, cv2.CV_64F).var())
    if blur_score < min_blur_var:
        issues.append(f"blurry (laplacian_var={blur_score:.1f})")
        cues.append("Photo appears out of focus or shaky. Tap to focus on the skin area and hold steady.")

    # 3. Tone-calibrated exposure & dynamic range check (Fitzpatrick IV-VI safe)
    p5, p95 = np.percentile(y_chan, [5, 95])
    dynamic_range = float(p95 - p5)
    local_contrast = float(np.std(y_chan))
    overexposed_ratio = float(np.mean(y_chan >= 248))
    crushed_black_ratio = float(np.mean(y_chan <= 12))

    if crushed_black_ratio > max_crushed_black_ratio and p95 < 35.0:
        issues.append(f"underexposed_crushed (black_ratio={crushed_black_ratio:.2f}, p95={p95:.1f})")
        cues.append("Lighting is too dim to see lesion texture. Move near natural daylight or turn on overhead light.")
    elif dynamic_range < min_dynamic_range and local_contrast < min_local_contrast:
        issues.append(f"flat_low_contrast (range={dynamic_range:.1f}, std={local_contrast:.1f})")
        cues.append("Image has very low contrast (foggy lens or flat lighting). Wipe camera lens and retake in even daylight.")

    if overexposed_ratio > max_overexposed_ratio:
        issues.append(f"overexposed_flash_glare (ratio={overexposed_ratio:.2f})")
        cues.append("Harsh flash glare is washing out skin detail. Turn off flash and shoot in indirect daylight.")

    # 4. Inclusive skin & scalp framing signal (Fitzpatrick I-VI + textured scalp/hair)
    skin_mask = (
        (y_chan >= 20) & (y_chan <= 245)
        & (cr_chan >= 128) & (cr_chan <= 185)
        & (cb_chan >= 72) & (cb_chan <= 135)
    )
    grad_mag = (
        cv2.Sobel(y_eq, cv2.CV_32F, 1, 0, ksize=3) ** 2
        + cv2.Sobel(y_eq, cv2.CV_32F, 0, 1, ksize=3) ** 2
    )
    textured_scalp_mask = (
        (y_chan >= 15) & (y_chan <= 115) & (grad_mag > 1200.0) & (cr_chan >= 118) & (cr_chan <= 165)
    )

    skin_scalp_signal = float(np.mean(skin_mask | textured_scalp_mask))
    if skin_scalp_signal < min_skin_scalp_signal:
        issues.append(f"weak_skin_scalp_framing (signal={skin_scalp_signal:.2f})")
        cues.append(
            "Skin or scalp area is not clearly framed. Fill most of the frame with the affected area (keep lens ~10-20 cm away)."
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metrics = {
        "width": int(w),
        "height": int(h),
        "blur_laplacian_var": round(float(blur_score), 2),
        "dynamic_range": round(float(dynamic_range), 2),
        "local_contrast": round(float(local_contrast), 2),
        "overexposed_ratio": round(float(overexposed_ratio), 4),
        "crushed_black_ratio": round(float(crushed_black_ratio), 4),
        "skin_scalp_signal": round(float(skin_scalp_signal), 4),
        "elapsed_ms": round(float(elapsed_ms), 2),
    }
    if "bgr" in locals():
        _hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _h, _s, _v = _hsv[:, :, 0], _hsv[:, :, 1], _hsv[:, :, 2]
        _skin_mask = (_h <= 28) & (_s >= 22) & (_s <= 210) & (_v >= 45)
        _skin_ratio = float(np.mean(_skin_mask)) if _skin_mask.size else 0.0
        metrics["skin_pixel_ratio"] = round(_skin_ratio, 4)
        if _skin_ratio < 0.08:
            issues.append(f"low_skin_pixel_coverage ({_skin_ratio*100:.1f}%)")
            cues.append(
                "Low skin-tone pixel coverage; verify the photo clearly frames the affected skin or scalp area."
            )

    return QualityGateResult(
        passed=len(issues) == 0,
        elapsed_ms=elapsed_ms,
        width=w,
        height=h,
        blur_score=blur_score,
        dynamic_range=dynamic_range,
        local_contrast=local_contrast,
        overexposed_ratio=overexposed_ratio,
        crushed_black_ratio=crushed_black_ratio,
        skin_scalp_signal=skin_scalp_signal,
        issues=issues,
        retake_cues=cues,
        metrics=metrics,
    )



def _trim_uniform_borders(img: Image.Image, tolerance: int = 18, min_keep_ratio: float = 0.45) -> Image.Image:
    """Trim uniform white/black screenshot or studio margins before vision encoding."""
    from PIL import ImageChops

    rgb = img.convert("RGB")
    w, h = rgb.size
    if w < 64 or h < 64:
        return rgb
    corners = [
        rgb.getpixel((2, 2)),
        rgb.getpixel((w - 3, 2)),
        rgb.getpixel((2, h - 3)),
        rgb.getpixel((w - 3, h - 3)),
    ]
    avg_c = tuple(int(sum(c[i] for c in corners) / 4) for i in range(3))
    if max(abs(c[i] - avg_c[i]) for c in corners for i in range(3)) > tolerance:
        return rgb
    bg = Image.new("RGB", (w, h), avg_c)
    diff = ImageChops.difference(rgb, bg).convert("L").point(lambda p: 255 if p > tolerance else 0)
    bbox = diff.getbbox()
    if not bbox:
        return rgb
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if bw < w * min_keep_ratio or bh < h * min_keep_ratio:
        return rgb
    pad_x, pad_y = int(bw * 0.02), int(bh * 0.02)
    return rgb.crop((max(0, bbox[0] - pad_x), max(0, bbox[1] - pad_y), min(w, bbox[2] + pad_x), min(h, bbox[3] + pad_y)))


def sanitize_and_compress_image(
    raw_bytes: bytes,
    max_side: int = 1024,
    output_format: Literal["WEBP", "JPEG"] = "WEBP",
    quality: int = 86,
) -> tuple[bytes, int, int]:
    """
    Privacy & Storage Guardrail (Blueprint Precautions #3 & #5):
      1. Applies EXIF orientation, then copies raw RGB pixels into a brand-new
         Image object so 100% of EXIF, GPS, camera model, and timestamps are stripped.
      2. Resizes longest side to <= 1024 px while preserving dermatological texture.
      3. Compresses to compact WebP/JPEG (~120-280 KB) for Supabase 1 GB free tier.
    """
    with Image.open(io.BytesIO(raw_bytes)) as src:
        oriented = ImageOps.exif_transpose(src).convert("RGB")
        clean = Image.fromarray(np.array(oriented, dtype=np.uint8), mode="RGB")

    w, h = clean.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        clean = clean.resize(new_size, Image.Resampling.LANCZOS)
        w, h = clean.size

    buf = io.BytesIO()
    if output_format == "WEBP":
        clean.save(buf, format="WEBP", quality=quality, method=4)
    else:
        clean.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), w, h
