"""
Nidaan-Rx v2.0 — Phase 2A Verification Suite (Quality Gate + EXIF Scrub + Extractor Specs).
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.vision.extractor import BACKBONE_SPECS
from app.vision.quality_gate import evaluate_image_quality, sanitize_and_compress_image


def _make_textured_skin_image(
    base_bgr: tuple[int, int, int],
    w: int = 640,
    h: int = 640,
    seed: int = 42,
) -> np.ndarray:
    """Synthesize a sharp skin patch with realistic pores/papules texture."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), base_bgr, dtype=np.float32)
    grain = rng.normal(0.0, 9.5, (h, w, 1)).astype(np.float32)
    img = np.clip(img + grain, 0, 255).astype(np.uint8)
    m = max(10, min(120, min(w, h) // 5))
    for _ in range(35):
        cx, cy = int(rng.integers(m, w - m)), int(rng.integers(m, h - m))
        r = int(rng.integers(4, 14))
        lesion_color = (
            max(0, base_bgr[0] - 18),
            max(0, base_bgr[1] - 25),
            min(255, base_bgr[2] + 22),
        )
        cv2.circle(img, (cx, cy), r, lesion_color, -1, cv2.LINE_AA)
    return img


def test_quality_gate_accepts_fitzpatrick_iv_to_vi_dark_skin_fast():
    for fitz_bgr in [(48, 72, 112), (32, 48, 78), (24, 36, 58)]:
        dark_skin = _make_textured_skin_image(fitz_bgr)
        res = evaluate_image_quality(dark_skin)
        assert res.passed is True, f"Dark skin {fitz_bgr} falsely rejected: {res.issues}"
        assert res.elapsed_ms < 50.0, f"Quality gate too slow: {res.elapsed_ms:.1f} ms"


def test_quality_gate_accepts_fitzpatrick_ii_to_iii_light_skin():
    light_skin = _make_textured_skin_image((155, 182, 215))
    res = evaluate_image_quality(light_skin)
    assert res.passed is True, f"Light skin rejected: {res.issues}"


def test_quality_gate_rejects_blurry_image():
    skin = _make_textured_skin_image((72, 102, 148))
    blurry = cv2.GaussianBlur(skin, (25, 25), 9.0)
    res = evaluate_image_quality(blurry)
    assert res.passed is False
    assert any("blurry" in i for i in res.issues)


def test_quality_gate_rejects_low_resolution():
    tiny = _make_textured_skin_image((72, 102, 148), w=160, h=160)
    res = evaluate_image_quality(tiny)
    assert res.passed is False
    assert any("low_resolution" in i for i in res.issues)


def test_quality_gate_rejects_harsh_flash_glare():
    blown_out = np.full((512, 512, 3), 253, dtype=np.uint8)
    res = evaluate_image_quality(blown_out)
    assert res.passed is False
    assert any("overexposed" in i for i in res.issues)


def test_quality_gate_rejects_pitch_black_lens_covered():
    black = np.full((512, 512, 3), 4, dtype=np.uint8)
    res = evaluate_image_quality(black)
    assert res.passed is False
    assert any("underexposed" in i for i in res.issues)


def test_quality_gate_rejects_foggy_flat_lens():
    foggy = np.full((512, 512, 3), (120, 125, 130), dtype=np.uint8)
    res = evaluate_image_quality(foggy)
    assert res.passed is False
    assert any("flat_low_contrast" in i for i in res.issues)


def test_quality_gate_rejects_non_skin_solid_blue_object():
    rng = np.random.default_rng(7)
    blue_obj = np.full((512, 512, 3), (210, 80, 20), dtype=np.float32)
    blue_obj = np.clip(blue_obj + rng.normal(0, 12, blue_obj.shape), 0, 255).astype(np.uint8)
    res = evaluate_image_quality(blue_obj)
    assert res.passed is False
    assert any("weak_skin_scalp_framing" in i for i in res.issues)


def test_sanitize_strips_exif_and_resizes_to_1024px():
    img = Image.fromarray(np.full((1800, 2400, 3), 120, dtype=np.uint8), mode="RGB")
    exif = img.getexif()
    exif[271] = "Apple"
    exif[272] = "iPhone 15"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    raw_with_exif = buf.getvalue()

    clean_bytes, cw, ch = sanitize_and_compress_image(raw_with_exif, max_side=1024, output_format="WEBP")
    assert max(cw, ch) == 1024
    assert len(clean_bytes) < 150_000

    reopened = Image.open(io.BytesIO(clean_bytes))
    assert len(reopened.getexif()) == 0


def test_backbone_specs_match_blueprint():
    assert BACKBONE_SPECS["panderm_base"].embed_dim == 768
    assert BACKBONE_SPECS["panderm_base"].license_id == "CC-BY-NC-ND-4.0"
    assert BACKBONE_SPECS["dinov2_small"].embed_dim == 384
    assert BACKBONE_SPECS["dinov2_small"].license_id == "Apache-2.0"