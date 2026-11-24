#!/usr/bin/env python3
"""
Nidaan-Rx Task 1 — Part 4: Physical Realism Augmenter & Exact Homography Warper.

Supports:
  - Full run / --resume across all 5,000 raw renders
  - Surgical re-augmentation via --only-tier tier3_hard or --ids rx_1239 rx_0375 ...
    (preserves untouched records in warped_boxes.jsonl and re-exports PaddleOCR labels)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Keep OpenCV single-threaded inside each worker process to avoid CPU thrashing on M2 Air
cv2.setNumThreads(1)

TAG_COLORS_BGR = {
    "BRAND": (38, 38, 220),       # Red
    "SALT": (235, 99, 37),        # Blue
    "DOSE": (74, 163, 22),        # Green
    "FREQ": (74, 163, 22),
    "INSTR": (74, 163, 22),
    "DURATION": (74, 163, 22),
    "QTY": (74, 163, 22),
    "FREQ_NOTE": (234, 51, 147),  # Purple
    "NOISE": (170, 170, 170),     # Grey
}

PEN_COLORS_BGR = [
    (65, 52, 195),    # Red ballpoint
    (118, 62, 188),   # Pink/magenta ballpoint
    (165, 72, 42),    # Blue ballpoint
    (52, 48, 175),    # Dark crimson pen
    (80, 60, 75),     # Dark purple-grey pen
]

DESK_PALETTES_BGR = [
    (42, 44, 48),     # Matte charcoal desk
    (68, 72, 76),     # Dark slate tabletop
    (52, 66, 86),     # Dark warm wood
    (85, 102, 122),   # Medium wooden desk
    (118, 120, 118),  # Grey office mat
    (32, 34, 38),     # Near-black desk pad
]


# ---------------------------------------------------------------------------
# Optional Augraphy Non-Geometric Ink Layer
# ---------------------------------------------------------------------------
def try_build_augraphy_ink_phase(seed: int):
    try:
        from augraphy import (  # type: ignore[import-not-found]
            AugraphyPipeline,
            DirtyDrum,
            InkBleed,
            SubtleNoise,
        )
        return AugraphyPipeline(
            ink_phase=[
                InkBleed(intensity_range=(0.04, 0.14), kernel_size=(3, 3), severity=(0.12, 0.28), p=0.45),
            ],
            paper_phase=[
                SubtleNoise(subtle_range=5, p=0.55),
                DirtyDrum(line_width_range=(1, 3), line_concentration=0.04, direction=0, p=0.25),
            ],
            post_phase=[],
            random_seed=seed,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page-Space Physical Artifacts (Pre-Warp)
# ---------------------------------------------------------------------------
def apply_paper_and_toner_texture(
    img: np.ndarray, tier: str, rng: random.Random, np_rng: np.random.Generator
) -> np.ndarray:
    """Add laser-printer micro-softness, warm/cool paper fiber grain, smudges, and creases."""
    h, w = img.shape[:2]
    out = img.copy()

    # 1. Subtle paper base tint
    if tier == "tier1_clean":
        paper_bgr = np.array([rng.uniform(246, 253), rng.uniform(247, 253), rng.uniform(246, 254)], dtype=np.float32)
    else:
        base = rng.uniform(234, 247)
        paper_bgr = np.array([
            base + rng.uniform(-5, 3),
            base + rng.uniform(-3, 4),
            base + rng.uniform(-2, 6),
        ], dtype=np.float32)

    out = np.clip(out.astype(np.float32) * (paper_bgr / 255.0), 0, 255).astype(np.uint8)

    # 2. Mild laser-toner softness (kept light in Tier 3 so post-warp camera blur never double-destroys small digits)
    if tier == "tier2_smartphone" and rng.random() < 0.50:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=rng.uniform(0.30, 0.55))
    elif tier == "tier3_hard" and rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=rng.uniform(0.25, 0.42))

    # 3. Fine paper grain
    grain_sigma = 1.5 if tier == "tier1_clean" else (3.2 if tier == "tier2_smartphone" else 4.0)
    gh, gw = max(64, h // 2), max(64, w // 2)
    grain = np_rng.normal(0.0, grain_sigma, (gh, gw, 1)).astype(np.float32)
    grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    out = np.clip(out.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # 4. Horizontal/diagonal paper fold crease
    crease_prob = 0.0 if tier == "tier1_clean" else (0.18 if tier == "tier2_smartphone" else 0.55)
    if rng.random() < crease_prob:
        n_creases = 1 if tier == "tier2_smartphone" else rng.randint(1, 2)
        for _ in range(n_creases):
            y_pos = int(rng.uniform(0.28, 0.72) * h)
            y_tilt = int(rng.uniform(-22, 22))
            band_h = rng.randint(18, 38)
            crease_map = np.zeros((h, w), dtype=np.float32)
            cv2.line(crease_map, (0, y_pos), (w - 1, y_pos + y_tilt), 1.0, thickness=band_h // 2)
            cv2.line(crease_map, (0, y_pos + band_h // 3), (w - 1, y_pos + y_tilt + band_h // 3), -0.6, thickness=band_h // 3)
            k = (band_h | 1) + 12
            if k % 2 == 0:
                k += 1
            crease_map = cv2.GaussianBlur(crease_map, (k, k), 0)
            strength = rng.uniform(8.0, 17.0) if tier == "tier3_hard" else rng.uniform(6.0, 12.0)
            out = np.clip(out.astype(np.float32) - crease_map[:, :, None] * strength, 0, 255).astype(np.uint8)

    # 5. Occasional tiny toner/finger smudge
    smudge_prob = 0.0 if tier == "tier1_clean" else (0.15 if tier == "tier2_smartphone" else 0.32)
    if rng.random() < smudge_prob:
        for _ in range(rng.randint(1, 2)):
            sx = int(rng.uniform(0.08, 0.88) * w)
            sy = int(rng.uniform(0.20, 0.85) * h)
            rx = rng.randint(8, 20)
            ry = rng.randint(6, 15)
            smudge = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(smudge, (sx, sy), (rx, ry), rng.uniform(0, 180), 0, 360, 1.0, -1)
            smudge = cv2.GaussianBlur(smudge, (21, 21), 0)
            alpha = rng.uniform(14.0, 32.0)
            out = np.clip(out.astype(np.float32) - smudge[:, :, None] * alpha, 0, 255).astype(np.uint8)

    return out


def _jitter_segment(p1: tuple[float, float], p2: tuple[float, float], rng: random.Random, steps: int = 6) -> np.ndarray:
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] * (1.0 - t) + p2[0] * t + rng.uniform(-0.85, 0.85)
        y = p1[1] * (1.0 - t) + p2[1] * t + rng.uniform(-0.85, 0.85)
        pts.append([int(round(x)), int(round(y))])
    return np.array(pts, dtype=np.int32)


def draw_pharmacist_marks(
    img: np.ndarray,
    row_anchors: list[dict[str, Any]],
    tier: str,
    rng: random.Random,
) -> int:
    """Draw realistic ballpoint pen ticks, crosses, or handwritten digits beside medicine names."""
    if not row_anchors or tier == "tier1_clean":
        return 0

    mark_doc_prob = 0.72 if tier == "tier2_smartphone" else 0.88
    if rng.random() > mark_doc_prob:
        return 0

    h, w = img.shape[:2]
    pen_color = rng.choice(PEN_COLORS_BGR)
    marks_drawn = 0

    for anc in row_anchors:
        if rng.random() > rng.uniform(0.65, 0.95):
            continue

        x1, y1, x2, y2 = anc["bbox"]
        cy = int(round((y1 + y2) * 0.5 + rng.uniform(-4, 6)))

        if tier == "tier3_hard" and rng.random() < 0.35:
            cx = int(round(x2 + rng.uniform(-8, 16)))
        else:
            cx = int(round(x2 + rng.uniform(14, 48)))

        cx = max(25, min(w - 30, cx))
        cy = max(25, min(h - 30, cy))

        mark_type = rng.choices(
            ["tick", "cross", "digit"],
            weights=[0.74, 0.14, 0.12] if tier == "tier2_smartphone" else [0.62, 0.20, 0.18],
        )[0]

        overlay = img.copy()
        scale = rng.uniform(0.85, 1.22)

        if mark_type == "tick":
            mw = rng.uniform(22, 36) * scale
            mh = rng.uniform(16, 26) * scale
            ang = math.radians(rng.uniform(-15, 15))
            ca, sa = math.cos(ang), math.sin(ang)
            raw_v = [(-mw * 0.45, rng.uniform(-2, 3)), (-mw * 0.10, mh * 0.45), (mw * 0.58, -mh * 0.52)]
            pts = [(cx + px * ca - py * sa, cy + px * sa + py * ca) for px, py in raw_v]
            th = max(2, int(round(rng.uniform(2.0, 3.4))))
            cv2.polylines(overlay, [_jitter_segment(pts[0], pts[1], rng, 5)], False, pen_color, th, cv2.LINE_AA)
            cv2.polylines(overlay, [_jitter_segment(pts[1], pts[2], rng, 7)], False, pen_color, max(2, th - 1), cv2.LINE_AA)
        elif mark_type == "cross":
            r = rng.uniform(8, 14) * scale
            th = max(2, int(round(rng.uniform(2.0, 3.0))))
            p1, p2 = (cx - r, cy - r), (cx + r, cy + r)
            p3, p4 = (cx + r, cy - r), (cx - r, cy + r)
            cv2.polylines(overlay, [_jitter_segment(p1, p2, rng, 4)], False, pen_color, th, cv2.LINE_AA)
            cv2.polylines(overlay, [_jitter_segment(p3, p4, rng, 4)], False, pen_color, th, cv2.LINE_AA)
        else:
            ch = rng.choice(["1", "2", "3", "P"])
            font = rng.choice([cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX])
            cv2.putText(
                overlay, ch, (int(cx - 8), int(cy + 10)),
                font, rng.uniform(0.75, 1.05) * scale, pen_color,
                max(2, int(round(rng.uniform(1.8, 2.8)))), cv2.LINE_AA,
            )

        alpha = rng.uniform(0.68, 0.88)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
        marks_drawn += 1

    return marks_drawn


def draw_punch_holes(
    img: np.ndarray,
    desk_color: tuple[int, int, int],
    tier: str,
    rng: random.Random,
) -> bool:
    """Draw 2 realistic left-margin binder punch holes with torn/shadowed rims."""
    if tier == "tier1_clean":
        return False
    prob = 0.65 if tier == "tier2_smartphone" else 0.78
    if rng.random() > prob:
        return False

    h, w = img.shape[:2]
    if tier == "tier3_hard" and rng.random() < 0.30:
        hx = int(rng.uniform(0.042, 0.068) * w)
    else:
        hx = int(rng.uniform(0.024, 0.048) * w)

    center_y = int(rng.uniform(0.44, 0.56) * h)
    half_gap = int(rng.uniform(0.125, 0.148) * h)
    radius = int(rng.uniform(14, 19))

    for hy in (center_y - half_gap, center_y + half_gap):
        if hy - radius < 10 or hy + radius >= h - 10:
            continue
        cv2.circle(img, (hx + 1, hy + 2), radius + 2, (115, 115, 112), -1, cv2.LINE_AA)
        hole_bgr = (
            max(10, desk_color[0] - rng.randint(15, 32)),
            max(10, desk_color[1] - rng.randint(15, 32)),
            max(10, desk_color[2] - rng.randint(15, 32)),
        )
        cv2.circle(img, (hx, hy), radius, hole_bgr, -1, cv2.LINE_AA)
        cv2.ellipse(img, (hx, hy), (radius, radius), 0, 200, 340, (165, 165, 160), 1, cv2.LINE_AA)

    return True


# ---------------------------------------------------------------------------
# Exact Homography Paper-on-Desk Compositing + BBox Warp
# ---------------------------------------------------------------------------
def build_desk_background(out_w: int, out_h: int, desk_color: tuple[int, int, int], np_rng: np.random.Generator) -> np.ndarray:
    bg = np.full((out_h, out_w, 3), desk_color, dtype=np.float32)
    lh, lw = max(64, out_h // 4), max(64, out_w // 4)
    noise = np_rng.normal(0.0, 4.5, (lh, lw, 1)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (1, 9), 0)
    noise = cv2.resize(noise, (out_w, out_h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return bg


def warp_paper_onto_desk(
    paper_img: np.ndarray,
    boxes: list[dict[str, Any]],
    tier: str,
    desk_color: tuple[int, int, int],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """
    Composite paper onto desk using an exact 3x3 homography M and warp all boxes via M.
    """
    h, w = paper_img.shape[:2]

    if tier == "tier1_clean":
        pad_l = rng.randint(0, 12)
        pad_r = rng.randint(0, 12)
        pad_t = rng.randint(0, 12)
        pad_b = rng.randint(0, 12)
        max_rot = 0.7
        persp_frac = 0.005
    elif tier == "tier2_smartphone":
        pad_l = rng.randint(18, 56)
        pad_r = rng.randint(18, 56)
        pad_t = rng.randint(16, 48)
        pad_b = rng.randint(20, 62)
        max_rot = 3.2
        persp_frac = 0.028
    else:  # tier3_hard
        pad_l = rng.randint(24, 70)
        pad_r = rng.randint(24, 70)
        pad_t = rng.randint(22, 62)
        pad_b = rng.randint(26, 76)
        max_rot = 6.0
        persp_frac = 0.048

    out_w = w + pad_l + pad_r
    out_h = h + pad_t + pad_b

    src_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    base_dst = np.array([
        [pad_l, pad_t],
        [pad_l + w - 1, pad_t],
        [pad_l + w - 1, pad_t + h - 1],
        [pad_l, pad_t + h - 1],
    ], dtype=np.float32)

    cx, cy = out_w * 0.5, out_h * 0.5
    angle_deg = rng.uniform(-max_rot, max_rot)
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)

    fit_scale = 1.0 - (abs(angle_deg) * 0.006 + persp_frac * 0.45)
    rot_dst = np.zeros_like(base_dst)
    for i in range(4):
        dx = (base_dst[i, 0] - cx) * fit_scale
        dy = (base_dst[i, 1] - cy) * fit_scale
        jx = rng.uniform(-persp_frac, persp_frac) * w * 0.52
        jy = rng.uniform(-persp_frac, persp_frac) * h * 0.42
        rot_dst[i, 0] = np.clip(cx + (dx * ca - dy * sa) + jx, 4, out_w - 5)
        rot_dst[i, 1] = np.clip(cy + (dx * sa + dy * ca) + jy, 4, out_h - 5)

    M = cv2.getPerspectiveTransform(src_corners, rot_dst)

    # INTER_CUBIC preserves sharper glyph edges on tiny decimal points and '%' signs
    warped_paper = cv2.warpPerspective(
        paper_img, M, (out_w, out_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    paper_mask = np.full((h, w), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        paper_mask, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    if tier == "tier1_clean" and pad_l == 0 and pad_r == 0:
        canvas = warped_paper
    else:
        bg = build_desk_background(out_w, out_h, desk_color, np_rng)
        shadow_mask = cv2.GaussianBlur(warped_mask, (21, 21), 0).astype(np.float32) / 255.0
        bg = np.clip(bg.astype(np.float32) * (1.0 - 0.35 * shadow_mask[:, :, None]), 0, 255).astype(np.uint8)
        alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
        canvas = np.clip(warped_paper.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)

    # Keep Tier 3 target width at 1240-1380 px so 13px salt text is never downscaled to illegible ~9px
    scale_factor = 1.0
    if tier == "tier2_smartphone":
        target_w = rng.randint(1160, 1340)
        scale_factor = target_w / float(out_w)
        new_h = int(round(out_h * scale_factor))
        canvas = cv2.resize(canvas, (target_w, new_h), interpolation=cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR)
        out_w, out_h = target_w, new_h
    elif tier == "tier3_hard":
        target_w = rng.randint(1240, 1380)
        scale_factor = target_w / float(out_w)
        new_h = int(round(out_h * scale_factor))
        canvas = cv2.resize(canvas, (target_w, new_h), interpolation=cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC)
        out_w, out_h = target_w, new_h

    warped_boxes: list[dict[str, Any]] = []
    if boxes:
        all_pts = np.array([b["points"] for b in boxes], dtype=np.float32)
        flat_pts = all_pts.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(flat_pts, M).reshape(-1, 4, 2) * scale_factor

        for b, quad in zip(boxes, transformed):
            pts_int = [
                [
                    int(np.clip(round(float(px)), 0, out_w - 1)),
                    int(np.clip(round(float(py)), 0, out_h - 1)),
                ]
                for px, py in quad
            ]
            warped_boxes.append({
                "transcription": b["transcription"],
                "points": pts_int,
                "tag": b["tag"],
                "row_index": b.get("row_index"),
            })

    geom_meta = {
        "rotation_deg": round(angle_deg, 2),
        "scale_factor": round(scale_factor, 4),
        "out_width": out_w,
        "out_height": out_h,
    }
    return canvas, warped_boxes, geom_meta


# ---------------------------------------------------------------------------
# Camera-Space Illumination, Blur & Sensor Noise (Post-Warp)
# ---------------------------------------------------------------------------
def apply_camera_lighting_and_optics(
    img: np.ndarray, tier: str, rng: random.Random, np_rng: np.random.Generator
) -> np.ndarray:
    """Apply overhead lighting gradient, phone cast shadow, white balance, realistic lens softness, and ISO noise."""
    h, w = img.shape[:2]
    lw, lh = max(32, w // 8), max(32, h // 8)
    yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    xn = xx / max(lw - 1, 1)
    yn = yy / max(lh - 1, 1)

    if tier == "tier1_clean":
        base_level = rng.uniform(0.95, 1.00)
        illum = base_level + rng.uniform(-0.03, 0.03) * (xn - 0.5) + rng.uniform(-0.03, 0.03) * (yn - 0.5)
        tint = np.array([rng.uniform(0.98, 1.0), rng.uniform(0.98, 1.0), rng.uniform(0.98, 1.0)], dtype=np.float32)
        min_illum = 0.85
    elif tier == "tier2_smartphone":
        base_level = rng.uniform(0.78, 0.91)
        illum = base_level + rng.uniform(-0.14, 0.14) * (xn - 0.5) + rng.uniform(-0.14, 0.14) * (yn - 0.5)
        cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.15, 0.75)
        illum *= 1.0 - rng.uniform(0.08, 0.18) * ((xn - cx) ** 2 + (yn - cy) ** 2)
        warm = rng.uniform(-0.04, 0.045)
        tint = np.array([1.0 - warm, 1.0 - abs(warm) * 0.2, 1.0 + warm], dtype=np.float32)
        min_illum = 0.58
    else:  # tier3_hard (realistic challenging lighting, still human-legible)
        base_level = rng.uniform(0.74, 0.88)
        illum = base_level + rng.uniform(-0.18, 0.18) * (xn - 0.5) + rng.uniform(-0.18, 0.18) * (yn - 0.5)
        if rng.random() < 0.75:
            ang = rng.uniform(0, 2 * math.pi)
            proj = (xn - rng.uniform(0.25, 0.75)) * math.cos(ang) + (yn - rng.uniform(0.25, 0.75)) * math.sin(ang)
            shadow_mask = 1.0 / (1.0 + np.exp(-proj * rng.uniform(7.0, 13.0)))
            illum *= 1.0 - rng.uniform(0.12, 0.22) * shadow_mask
        warm = rng.uniform(-0.05, 0.06)
        tint = np.array([1.0 - warm, 1.0 - abs(warm) * 0.2, 1.0 + warm], dtype=np.float32)
        min_illum = 0.55

    illum_full = cv2.resize(np.clip(illum, min_illum, 1.04), (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.clip(img.astype(np.float32) * illum_full[:, :, None] * tint[None, None, :], 0, 255).astype(np.uint8)

    if tier == "tier2_smartphone" and rng.random() < 0.28:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=rng.uniform(0.32, 0.58))
    elif tier == "tier3_hard":
        # Mild handheld tremor (weighted 3x3) or moderate defocus — never destructive 5x5 smear
        roll = rng.random()
        if roll < 0.38:
            side = rng.uniform(0.16, 0.24)
            center = 1.0 - 2.0 * side
            kernel = np.zeros((3, 3), dtype=np.float32)
            orient = rng.choice(["horiz", "diag", "anti"])
            if orient == "horiz":
                kernel[1, 0], kernel[1, 1], kernel[1, 2] = side, center, side
            elif orient == "diag":
                kernel[0, 0], kernel[1, 1], kernel[2, 2] = side, center, side
            else:
                kernel[0, 2], kernel[1, 1], kernel[2, 0] = side, center, side
            out = cv2.filter2D(out, -1, kernel)
        elif roll < 0.76:
            out = cv2.GaussianBlur(out, (3, 3), sigmaX=rng.uniform(0.42, 0.68))

    iso_sigma = 0.0 if tier == "tier1_clean" else (2.0 if tier == "tier2_smartphone" else 2.8)
    if iso_sigma > 0:
        noise = np_rng.normal(0.0, iso_sigma, (h, w, 3)).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def save_debug_warped_overlay(img: np.ndarray, warped_boxes: list[dict[str, Any]], out_path: Path) -> None:
    vis = img.copy()
    for b in warped_boxes:
        pts = np.array(b["points"], dtype=np.int32).reshape((-1, 1, 2))
        color = TAG_COLORS_BGR.get(b["tag"], (11, 158, 245))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# ---------------------------------------------------------------------------
# Worker Function (One Image End-to-End)
# ---------------------------------------------------------------------------
def process_single_record(task: tuple[dict[str, Any], str, int, bool, bool]) -> dict[str, Any]:
    rec, out_dir_str, base_seed, use_augraphy, save_vis = task
    out_dir = Path(out_dir_str)
    image_id = rec["image_id"]
    tier = rec["difficulty"]

    try:
        numeric_id = int(image_id.split("_")[-1])
    except ValueError:
        numeric_id = abs(hash(image_id)) % 100000
    img_seed = base_seed * 100000 + numeric_id
    rng = random.Random(img_seed)
    np_rng = np.random.default_rng(img_seed)

    raw_png = out_dir / rec["raw_image_path"]
    img = cv2.imread(str(raw_png), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"Could not read {raw_png}"}

    used_augraphy = False
    if use_augraphy and tier != "tier1_clean":
        pipe = try_build_augraphy_ink_phase(img_seed)
        if pipe is not None:
            try:
                img = pipe(img)
                used_augraphy = True
            except Exception:
                pass

    desk_color = rng.choice(DESK_PALETTES_BGR)
    img = apply_paper_and_toner_texture(img, tier, rng, np_rng)
    pen_marks = draw_pharmacist_marks(img, rec.get("row_anchors", []), tier, rng)
    has_holes = draw_punch_holes(img, desk_color, tier, rng)

    canvas, warped_boxes, geom_meta = warp_paper_onto_desk(
        img, rec.get("boxes", []), tier, desk_color, rng, np_rng
    )

    final_img = apply_camera_lighting_and_optics(canvas, tier, rng, np_rng)

    if tier == "tier1_clean":
        jpeg_q = rng.randint(88, 95)
    elif tier == "tier2_smartphone":
        jpeg_q = rng.randint(82, 91)
    else:
        jpeg_q = rng.randint(78, 88)

    out_jpg = out_dir / rec["file_name"]
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_jpg), final_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    if not ok:
        return {"error": f"Failed to write {out_jpg}"}

    if save_vis:
        vis_path = out_dir / "debug_aug_boxes" / f"aug_{tier}_{image_id}.jpg"
        save_debug_warped_overlay(final_img, warped_boxes, vis_path)

    return {
        "image_id": image_id,
        "file_name": rec["file_name"],
        "split": rec["split"],
        "template_id": rec["template_id"],
        "language": rec["language"],
        "difficulty": tier,
        "is_hard_negative": rec["is_hard_negative"],
        "width": geom_meta["out_width"],
        "height": geom_meta["out_height"],
        "boxes": warped_boxes,
        "augmentation_meta": {
            **geom_meta,
            "pen_marks": pen_marks,
            "punch_holes": has_holes,
            "jpeg_quality": jpeg_q,
            "used_augraphy": used_augraphy,
        },
    }


# ---------------------------------------------------------------------------
# Manifest Loading, Atomic Rewrite & PaddleOCR Format 2 Export
# ---------------------------------------------------------------------------
def load_raw_box_records(raw_boxes_path: Path, limit: int | None) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with raw_boxes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_id[rec["image_id"]] = rec
    records = [by_id[k] for k in sorted(by_id.keys())]
    if limit:
        records = records[:limit]
    return records


def load_completed_warped(warped_path: Path, out_dir: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not warped_path.exists():
        return done
    with warped_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                img_id = rec.get("image_id")
                fname = rec.get("file_name")
                if img_id and fname and (out_dir / fname).exists():
                    done[img_id] = rec
            except json.JSONDecodeError:
                continue
    return done


def write_warped_manifest_sorted(records_by_id: dict[str, dict[str, Any]], warped_path: Path) -> None:
    tmp_path = warped_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for img_id in sorted(records_by_id.keys()):
            f.write(json.dumps(records_by_id[img_id], ensure_ascii=False) + "\n")
    tmp_path.replace(warped_path)


def export_paddleocr_labels(records_by_id: dict[str, dict[str, Any]], out_dir: Path) -> None:
    """Write native PaddleOCR Format 2 tab-separated train/val/test label files."""
    lbl_dir = out_dir / "paddleocr_labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    handles = {
        sp: (lbl_dir / f"{sp}_det_rec.txt").open("w", encoding="utf-8")
        for sp in ("train", "val", "test")
    }
    counts = {"train": 0, "val": 0, "test": 0}
    try:
        for img_id in sorted(records_by_id.keys()):
            rec = records_by_id[img_id]
            sp = rec["split"]
            if sp not in handles:
                continue
            paddle_items = [
                {"transcription": b["transcription"], "points": b["points"]}
                for b in rec["boxes"]
            ]
            line = f"{rec['file_name']}\t{json.dumps(paddle_items, ensure_ascii=False)}\n"
            handles[sp].write(line)
            counts[sp] += 1
    finally:
        for h in handles.values():
            h.close()

    print(f"  PaddleOCR labels: {lbl_dir} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 4: Physical Realism Augmenter & Exact Homography Warper")
    parser.add_argument("--out-dir", type=Path, default=Path("synthetic_rx_dataset"))
    parser.add_argument("--workers", type=int, default=3, help="Parallel worker processes (2-3 recommended for 8 GB M2 Air)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic augmentations")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matching images")
    parser.add_argument("--only-tier", choices=["tier1_clean", "tier2_smartphone", "tier3_hard"], default=None,
                        help="Surgically re-augment only images in this tier and merge back into manifests")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Surgically re-augment only specific image IDs (e.g. --ids rx_1239 rx_0375 rx_4037 rx_4853)")
    parser.add_argument("--debug-vis", type=int, default=0, help="Save N warped bounding-box overlay JPEGs")
    parser.add_argument("--resume", action="store_true", help="Skip already-augmented images cleanly")
    parser.add_argument("--use-augraphy", action="store_true", help="Optionally layer Augraphy InkBleed/DirtyDrum if installed")
    parser.add_argument("--log-every", type=int, default=100, help="Progress log interval")
    args = parser.parse_args()

    raw_boxes_path = args.out_dir / "manifests" / "raw_boxes.jsonl"
    warped_boxes_path = args.out_dir / "manifests" / "warped_boxes.jsonl"

    if not raw_boxes_path.exists():
        raise FileNotFoundError(f"Missing {raw_boxes_path}. Run Part 3 (renderer.py) first.")

    surgical_mode = bool(args.only_tier or args.ids)
    raw_records = load_raw_box_records(raw_boxes_path, None if surgical_mode else args.limit)
    completed = load_completed_warped(warped_boxes_path, args.out_dir) if (args.resume or surgical_mode) else {}

    if surgical_mode:
        target_ids = set(args.ids) if args.ids else None
        pending = [
            r for r in raw_records
            if (args.only_tier is None or r["difficulty"] == args.only_tier)
            and (target_ids is None or r["image_id"] in target_ids)
        ]
        if args.limit:
            pending = pending[:args.limit]
    else:
        pending = [r for r in raw_records if r["image_id"] not in completed]

    vis_ids: set[str] = set()
    if args.debug_vis > 0:
        if surgical_mode:
            vis_ids = {r["image_id"] for r in pending[:args.debug_vis]}
        else:
            per_tier = max(1, math.ceil(args.debug_vis / 3))
            tier_counts = {"tier1_clean": 0, "tier2_smartphone": 0, "tier3_hard": 0}
            for r in pending:
                t = r["difficulty"]
                if t in tier_counts and tier_counts[t] < per_tier and len(vis_ids) < args.debug_vis:
                    vis_ids.add(r["image_id"])
                    tier_counts[t] += 1

    if not pending:
        print("[INFO] No matching pending images to augment.")
        export_paddleocr_labels(completed, args.out_dir)
        return

    mode_desc = f"surgical ({args.only_tier or 'ids'})" if surgical_mode else "standard"
    print(f"[INFO] Augmenting {len(pending):,} images [{mode_desc}] with {args.workers} workers...")

    tasks = [
        (r, str(args.out_dir), args.seed, args.use_augraphy, r["image_id"] in vis_ids)
        for r in pending
    ]

    start_t = time.time()
    processed = 0
    errors: list[str] = []
    all_warped = dict(completed)

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process_single_record, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if "error" in res:
                errors.append(res["error"])
                print(f"[WARN] {res['error']}")
                continue

            all_warped[res["image_id"]] = res
            processed += 1

            if processed % args.log_every == 0 or processed == len(pending):
                elapsed = time.time() - start_t
                rate = processed / max(elapsed, 0.001)
                eta = (len(pending) - processed) / max(rate, 0.001)
                print(
                    f"  [{processed:>4}/{len(pending)}] "
                    f"{rate:5.1f} img/s | elapsed {elapsed:5.1f}s | ETA {eta:5.1f}s"
                )

    write_warped_manifest_sorted(all_warped, warped_boxes_path)

    total_t = time.time() - start_t
    print(f"\n[DONE] Augmented {processed:,} JPEGs in {total_t:.1f}s ({len(errors)} errors).")
    print(f"  Final JPEGs     : {args.out_dir / 'images'}")
    print(f"  Warped manifest : {warped_boxes_path} ({len(all_warped):,} total records)")
    if vis_ids:
        print(f"  Debug overlays  : {args.out_dir / 'debug_aug_boxes'} ({len(vis_ids)} images)")
    export_paddleocr_labels(all_warped, args.out_dir)


if __name__ == "__main__":
    main()
