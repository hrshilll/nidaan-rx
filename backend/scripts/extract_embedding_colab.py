#!/usr/bin/env python3
"""Nidaan-Rx v2.0 — Phase 2B (Streaming Low-RAM) Embedding Extractor."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.vision.quality_gate import evaluate_image_quality
from app.vision.taxonomy_mapper import (
    TABULAR_DIM,
    TIER1_LIST,
    TIER1_TO_IDX,
    encode_scin_tabular_features,
    map_raw_label,
)


class ColabVisionEncoder:
    def __init__(self, backbone: str, panderm_ckpt: str | None = None, device: str | None = None) -> None:
        self.backbone = backbone
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.preprocess = None
        self.model = None
        self.mode = None

        if backbone == "panderm_base" and not panderm_ckpt:
            import open_clip

            hub_id = "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256"
            print(f"[INFO] Loading PanDerm-Base via OpenCLIP ({hub_id})...", flush=True)
            model, _, preprocess = open_clip.create_model_and_transforms(hub_id)
            self.model = model.to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.preprocess = preprocess
            self.mode = "open_clip"
        elif backbone == "panderm_base" and panderm_ckpt:
            from app.vision.extractor import FrozenFeatureExtractor

            self.ext = FrozenFeatureExtractor("panderm_base", checkpoint_path=panderm_ckpt, device=str(self.device))
            self.mode = "local_ext"
        elif backbone in {"dinov2_small", "dinov2_base"}:
            from app.vision.extractor import FrozenFeatureExtractor

            self.ext = FrozenFeatureExtractor(backbone, device=str(self.device))
            self.mode = "local_ext"
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        self.embed_dim = int(self.encode_pil_batch([dummy]).shape[-1])
        print(f"[INFO] Backbone '{backbone}' ready on {self.device} | embed_dim={self.embed_dim}", flush=True)

    @torch.inference_mode()
    def encode_pil_batch(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        if not images:
            return np.empty((0, self.embed_dim), dtype=np.float16)

        use_amp = self.device.type == "cuda"
        out_chunks: list[np.ndarray] = []

        if self.mode == "local_ext":
            for i in range(0, len(images), batch_size):
                chunk = images[i : i + batch_size]
                batch = torch.stack([self.ext.preprocess(im) for im in chunk], dim=0).to(self.device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    feats = self.ext.model(batch)
                    if feats.ndim == 3:
                        feats = feats[:, 0, :]
                feats = F.normalize(feats.float(), p=2, dim=-1)
                out_chunks.append(feats.cpu().numpy().astype(np.float16))
            return np.concatenate(out_chunks, axis=0)

        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            tensors = torch.stack([self.preprocess(im) for im in chunk], dim=0).to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                feats = self.model.encode_image(tensors)
            feats = F.normalize(feats.float(), p=2, dim=-1)
            out_chunks.append(feats.cpu().numpy().astype(np.float16))
        return np.concatenate(out_chunks, axis=0)


def _prep_rgb_thumb(im: Image.Image, max_side: int = 512) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.BILINEAR)
    return im


def _parse_scin_top_label(raw_weighted: Any) -> str | None:
    if raw_weighted is None:
        return None
    obj = raw_weighted
    if isinstance(raw_weighted, str):
        s = raw_weighted.strip()
        if not s or s == "{}":
            return None
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                return None
    if isinstance(obj, dict) and obj:
        return max(obj.items(), key=lambda kv: float(kv[1]))[0]
    return None


def iter_scin_cases(max_cases: int | None = None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    print("\n[INFO] Streaming google/scin ...", flush=True)
    ds = load_dataset("google/scin", split="train")
    img_cols = [c for c in ("image_1_path", "image_2_path", "image_3_path") if c in ds.column_names]
    yielded, total_rows, t0 = 0, len(ds), time.time()

    for idx, row in enumerate(ds):
        if max_cases and yielded >= max_cases:
            break
        top_label = _parse_scin_top_label(row.get("weighted_skin_condition_label"))
        if not top_label:
            names = row.get("dermatologist_skin_condition_label_name")
            if isinstance(names, list) and names:
                top_label = str(names[0])
        if not top_label:
            continue
        mapped = map_raw_label(top_label)
        if mapped is None:
            continue
        fam, canonical_id, tier2_lbl = mapped
        pil_views = [_prep_rgb_thumb(row[c], 512) for c in img_cols if isinstance(row.get(c), Image.Image)]
        if not pil_views:
            continue
        yielded += 1
        if yielded % 250 == 0:
            print(f"  [SCIN] {yielded:,} mapped (row {idx+1:,}/{total_rows:,}) | {yielded/max(time.time()-t0,0.1):.1f} cases/s", flush=True)
        yield {
            "case_id": f"scin_{row.get('case_id', idx)}",
            "source": "scin",
            "images": pil_views[:3],
            "tier1_family": fam.value,
            "tier1_idx": TIER1_TO_IDX[fam],
            "canonical_id": canonical_id,
            "tier2_label": tier2_lbl,
            "tabular": encode_scin_tabular_features(row),
            "has_tabular": True,
        }
    print(f"  -> Finished SCIN: {yielded:,} cases in {time.time()-t0:.1f}s", flush=True)
    del ds
    gc.collect()


def iter_fitzpatrick17k_clean_cases(max_cases: int | None = None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    bad_hashes: set[str] = set()
    try:
        cp_ds = load_dataset("Digital-Dermatology/CleanPatrick", "Combined Annotations", split="test")
        for r in cp_ds:
            md5 = str(r.get("md5hash") or r.get("image_id") or "")
            if md5 and any(bool(r.get(k)) for k in ("is_off_topic", "off_topic", "is_duplicate", "near_duplicate", "label_error") if k in r):
                bad_hashes.add(md5)
        print(f"\n[INFO] CleanPatrick flagged {len(bad_hashes):,} noisy/duplicate samples", flush=True)
        del cp_ds
    except Exception as e:
        print(f"\n[WARN] CleanPatrick unavailable ({e}); using raw Fitzpatrick17k.", flush=True)

    print("[INFO] Streaming spycoder/fitzpatrick ...", flush=True)
    ds = load_dataset("spycoder/fitzpatrick", split="train")
    yielded, total_rows, t0 = 0, len(ds), time.time()

    for idx, row in enumerate(ds):
        if max_cases and yielded >= max_cases:
            break
        md5 = str(row.get("md5hash") or idx)
        if md5 in bad_hashes:
            continue
        mapped = map_raw_label(str(row.get("label") or row.get("three_partition_label") or ""))
        if mapped is None or not isinstance(row.get("image"), Image.Image):
            continue
        fam, canonical_id, tier2_lbl = mapped
        tab = np.zeros(TABULAR_DIM, dtype=np.float32)
        try:
            fst_f = float(row.get("fitzpatrick_scale") or row.get("fitzpatrick_centaur"))
            tab[3] = (fst_f - 1.0) / 5.0 if 1.0 <= fst_f <= 6.0 else 0.5
        except (TypeError, ValueError):
            tab[3] = 0.5
        yielded += 1
        if yielded % 1000 == 0:
            print(f"  [F17K-C] {yielded:,} mapped (row {idx+1:,}/{total_rows:,}) | {yielded/max(time.time()-t0,0.1):.1f} cases/s", flush=True)
        yield {
            "case_id": f"f17k_{md5}",
            "source": "fitzpatrick17k_c",
            "images": [_prep_rgb_thumb(row["image"], 512)],
            "tier1_family": fam.value,
            "tier1_idx": TIER1_TO_IDX[fam],
            "canonical_id": canonical_id,
            "tier2_label": tier2_lbl,
            "tabular": tab,
            "has_tabular": False,
        }
    print(f"  -> Finished Fitzpatrick17k-C: {yielded:,} cases in {time.time()-t0:.1f}s", flush=True)
    del ds
    gc.collect()


def iter_indian_derm_hf_cases(max_cases: int | None = None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    print("\n[INFO] Streaming actorcritic/DermaConIN_SkinDisNet_ArsenicSkinImageBD ...", flush=True)
    ds = load_dataset("actorcritic/DermaConIN_SkinDisNet_ArsenicSkinImageBD", split="train")
    label_names = ds.features["label"].names if hasattr(ds.features.get("label"), "names") else None
    yielded, total_rows, t0 = 0, len(ds), time.time()

    for idx, row in enumerate(ds):
        if max_cases and yielded >= max_cases:
            break
        raw_lbl = label_names[row["label"]] if label_names is not None else str(row.get("label", ""))
        if str(raw_lbl).strip().lower() in {"others", "other", "healthy", "normal"}:
            continue
        mapped = map_raw_label(str(raw_lbl))
        if mapped is None or not isinstance(row.get("image"), Image.Image):
            continue
        fam, canonical_id, tier2_lbl = mapped
        tab = np.zeros(TABULAR_DIM, dtype=np.float32)
        tab[3], tab[4] = 0.65, 0.60
        yielded += 1
        if yielded % 1000 == 0:
            print(f"  [IndianDerm] {yielded:,} mapped (row {idx+1:,}/{total_rows:,}) | {yielded/max(time.time()-t0,0.1):.1f} cases/s", flush=True)
        yield {
            "case_id": f"inderm_{idx}",
            "source": "dermacon_in_hf",
            "images": [_prep_rgb_thumb(row["image"], 512)],
            "tier1_family": fam.value,
            "tier1_idx": TIER1_TO_IDX[fam],
            "canonical_id": canonical_id,
            "tier2_label": tier2_lbl,
            "tabular": tab,
            "has_tabular": False,
        }
    print(f"  -> Finished IndianDerm: {yielded:,} cases in {time.time()-t0:.1f}s", flush=True)
    del ds
    gc.collect()


def assign_stratified_splits(meta_records: list[dict[str, Any]], seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    by_strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, c in enumerate(meta_records):
        by_strata[(c["source"], c["tier1_idx"])].append(i)
    splits = ["train"] * len(meta_records)
    for idxs in by_strata.values():
        arr = np.array(idxs, dtype=np.int64)
        rng.shuffle(arr)
        n = len(arr)
        n_test = max(1, int(round(0.15 * n))) if n >= 5 else 0
        n_val = max(1, int(round(0.15 * n))) if n >= 4 else 0
        for j in arr[:n_test]:
            splits[int(j)] = "test"
        for j in arr[n_test : n_test + n_val]:
            splits[int(j)] = "val"
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Nidaan-Rx v2.0 Phase 2B Streaming Embedding Extractor")
    parser.add_argument("--out-dir", type=Path, default=Path("nidaan_v2_embeddings"))
    parser.add_argument("--backbones", nargs="+", default=["panderm_base", "dinov2_small"],
                        choices=["panderm_base", "dinov2_small", "dinov2_base"])
    parser.add_argument("--panderm-ckpt", type=str, default=None)
    parser.add_argument("--no-scin", action="store_true")
    parser.add_argument("--no-f17k", action="store_true")
    parser.add_argument("--no-indian-hf", action="store_true")
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument("--case-batch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-tier2-count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0, max_views = time.time(), 3
    encoders = {bb: ColabVisionEncoder(bb, panderm_ckpt=args.panderm_ckpt) for bb in args.backbones}
    bag_embs_chunks: dict[str, list[np.ndarray]] = {bb: [] for bb in args.backbones}
    bag_mask_chunks: list[np.ndarray] = []
    meta_records: list[dict[str, Any]] = []
    qg_checked = qg_passed = total_views = 0

    def flush(buf: list[dict[str, Any]]) -> None:
        nonlocal qg_checked, qg_passed, total_views
        if not buf:
            return
        flat_imgs: list[Image.Image] = []
        slices: list[tuple[int, int]] = []
        for c in buf:
            s = len(flat_imgs)
            flat_imgs.extend(c["images"][:max_views])
            slices.append((s, len(flat_imgs)))
        total_views += len(flat_imgs)

        for im in flat_imgs:
            if qg_checked < 300 and (total_views % 37 == 0 or qg_checked < 20):
                qg_checked += 1
                if evaluate_image_quality(im).passed:
                    qg_passed += 1

        b_mask = np.zeros((len(buf), max_views), dtype=np.bool_)
        for i, (s, e) in enumerate(slices):
            b_mask[i, : (e - s)] = True
        bag_mask_chunks.append(b_mask)

        for bb, enc in encoders.items():
            flat_vecs = enc.encode_pil_batch(flat_imgs, batch_size=args.batch_size)
            b_embs = np.zeros((len(buf), max_views, enc.embed_dim), dtype=np.float16)
            for i, (s, e) in enumerate(slices):
                b_embs[i, : (e - s), :] = flat_vecs[s:e]
            bag_embs_chunks[bb].append(b_embs)

        for c in buf:
            meta_records.append({
                "case_id": c["case_id"],
                "source": c["source"],
                "tier1_family": c["tier1_family"],
                "tier1_idx": c["tier1_idx"],
                "canonical_id": c["canonical_id"],
                "tier2_label": c["tier2_label"],
                "tabular": c["tabular"],
                "has_tabular": c["has_tabular"],
                "view_count": len(c["images"][:max_views]),
            })
        buf.clear()

    gens = []
    if not args.no_scin:
        gens.append(iter_scin_cases(args.max_per_source))
    if not args.no_f17k:
        gens.append(iter_fitzpatrick17k_clean_cases(args.max_per_source))
    if not args.no_indian_hf:
        gens.append(iter_indian_derm_hf_cases(args.max_per_source))

    buf: list[dict[str, Any]] = []
    for g in gens:
        for case in g:
            buf.append(case)
            if len(buf) >= args.case_batch:
                flush(buf)
        flush(buf)
        gc.collect()

    if not meta_records:
        raise RuntimeError("No cases extracted.")

    raw_t2_counts = Counter(c["tier2_label"] for c in meta_records)
    for c in meta_records:
        c["tier2_final"] = c["canonical_id"] if raw_t2_counts[c["tier2_label"]] < args.min_tier2_count else c["tier2_label"]

    tier2_vocab = sorted({c["tier2_final"] for c in meta_records})
    tier2_to_idx = {lbl: i for i, lbl in enumerate(tier2_vocab)}
    canonical_vocab = sorted({c["canonical_id"] for c in meta_records})
    canonical_to_idx = {lbl: i for i, lbl in enumerate(canonical_vocab)}
    splits = assign_stratified_splits(meta_records, seed=args.seed)
    split_code_map = {"train": 0, "val": 1, "test": 2}

    bag_mask = np.concatenate(bag_mask_chunks, axis=0)
    view_counts_arr = np.array([c["view_count"] for c in meta_records], dtype=np.int8)
    tabular_arr = np.stack([c["tabular"] for c in meta_records], axis=0).astype(np.float32)
    has_tabular_arr = np.array([c["has_tabular"] for c in meta_records], dtype=np.bool_)
    tier1_arr = np.array([c["tier1_idx"] for c in meta_records], dtype=np.int64)
    tier2_arr = np.array([tier2_to_idx[c["tier2_final"]] for c in meta_records], dtype=np.int64)
    canonical_arr = np.array([canonical_to_idx[c["canonical_id"]] for c in meta_records], dtype=np.int64)
    split_arr = np.array([split_code_map[s] for s in splits], dtype=np.int8)
    case_ids_arr = np.array([c["case_id"] for c in meta_records])
    sources_arr = np.array([c["source"] for c in meta_records])

    for bb in args.backbones:
        bag_embs = np.concatenate(bag_embs_chunks[bb], axis=0)
        out_npz = args.out_dir / f"embeddings_{bb}.npz"
        np.savez_compressed(
            out_npz,
            embeddings=bag_embs,
            view_mask=bag_mask,
            view_counts=view_counts_arr,
            tabular=tabular_arr,
            has_tabular=has_tabular_arr,
            tier1_y=tier1_arr,
            tier2_y=tier2_arr,
            canonical_y=canonical_arr,
            split=split_arr,
            case_ids=case_ids_arr,
            sources=sources_arr,
        )
        print(f"  -> Saved {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB) | shape={bag_embs.shape}", flush=True)

    manifest = {
        "total_cases": len(meta_records),
        "total_views": total_views,
        "multi_view_cases": int(np.sum(view_counts_arr > 1)),
        "sources": dict(Counter(c["source"] for c in meta_records)),
        "splits": dict(Counter(splits)),
        "tier1_classes": [f.value for f in TIER1_LIST],
        "tier1_counts": {TIER1_LIST[k].value: int(v) for k, v in Counter(tier1_arr.tolist()).items()},
        "canonical_classes": canonical_vocab,
        "tier2_classes": tier2_vocab,
        "tier2_to_tier1": {c["tier2_final"]: c["tier1_idx"] for c in meta_records},
        "tier2_to_canonical": {c["tier2_final"]: c["canonical_id"] for c in meta_records},
        "tabular_dim": TABULAR_DIM,
        "quality_gate_sample_pass_rate": round(qg_passed / max(1, qg_checked), 4),
        "backbones_extracted": args.backbones,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    manifest_path = args.out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] {manifest['elapsed_sec']}s | {manifest['total_cases']:,} cases | Tier-2: {len(tier2_vocab)}", flush=True)
    print(f"  Sources: {manifest['sources']} | Splits: {manifest['splits']}", flush=True)


if __name__ == "__main__":
    main()