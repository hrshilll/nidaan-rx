#!/usr/bin/env python3
"""
Nidaan-Rx v2.0 — Phase 3: 4-Stage Ablation Trainer & Model Calibration (Local M2 MPS).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.fusion_net import (
    AblationStage,
    JointHierarchicalLoss,
    MahalanobisOODDetector,
    NidaanMultimodalNet,
)
from app.vision.extractor import resolve_device

STAGES: list[AblationStage] = [
    "stage1_linear_baseline",
    "stage2_attention_mil",
    "stage3_adapter_hierarchical_focal",
    "stage4_multimodal_fusion",
]


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: list[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        denom = 2.0 * tp + fp + fn
        if denom > 0:
            f1s.append((2.0 * tp) / denom)
    return float(np.mean(f1s)) if f1s else 0.0


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_bounds[i]) & (confidences <= bin_bounds[i + 1])
        prop = float(np.mean(in_bin))
        if prop > 0:
            ece += abs(float(np.mean(accuracies[in_bin])) - float(np.mean(confidences[in_bin]))) * prop
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.detach().clone()
    log_temp = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=0.1, max=8.0)
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).clamp(min=0.1, max=8.0).item())


def compute_inverse_freq_weights(labels: np.ndarray, num_classes: int, power: float = 0.45) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / counts) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_split(
    model: NidaanMultimodalNet,
    loader: DataLoader,
    device: torch.device,
    num_tier1: int,
    num_canonical: int,
) -> dict[str, Any]:
    model.eval()
    all_t1_logits, all_can_logits = [], []
    all_t1_y, all_can_y, all_ht = [], [], []
    all_reps = []

    for emb, mask, tab, ht, t1_y, _, can_y in loader:
        emb, mask, tab, ht = emb.to(device), mask.to(device), tab.to(device), ht.to(device)
        out = model(emb, mask, tab, has_tabular=ht)
        all_ht.append(ht.cpu().numpy())
        all_t1_logits.append(out["logits_tier1"].cpu())
        all_can_logits.append(out["logits_canonical"].cpu())
        all_reps.append(out["rep"].cpu().numpy())
        all_t1_y.append(t1_y.numpy())
        all_can_y.append(can_y.numpy())

    t1_logits = torch.cat(all_t1_logits, dim=0)
    can_logits = torch.cat(all_can_logits, dim=0)
    t1_y = np.concatenate(all_t1_y, axis=0)
    can_y = np.concatenate(all_can_y, axis=0)
    reps = np.concatenate(all_reps, axis=0)

    t1_probs = torch.softmax(t1_logits, dim=-1).numpy()
    can_probs = torch.softmax(can_logits, dim=-1).numpy()

    t1_pred = t1_probs.argmax(axis=1)
    can_pred = can_probs.argmax(axis=1)
    top3_hits = sum(
        1 for i, y in enumerate(can_y) if int(y) in np.argsort(can_probs[i])[-3:]
    )

    return {
        "t1_logits": t1_logits,
        "can_logits": can_logits,
        "t1_y": t1_y,
        "can_y": can_y,
        "reps": reps,
        "tier1_acc": round(float(np.mean(t1_pred == t1_y)), 4),
        "tier1_macro_f1": round(compute_macro_f1(t1_y, t1_pred, num_tier1), 4),
        "canonical_top1_acc": round(float(np.mean(can_pred == can_y)), 4),
        "canonical_top3_acc": round(top3_hits / max(1, len(can_y)), 4),
        "canonical_macro_f1": round(compute_macro_f1(can_y, can_pred, num_canonical), 4),
        "canonical_ece": round(compute_ece(can_probs, can_y), 4),
    }


def train_single_stage(
    stage: AblationStage,
    data: dict[str, np.ndarray],
    manifest: dict[str, Any],
    device: torch.device,
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[NidaanMultimodalNet, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    emb = data["embeddings"].astype(np.float32)
    mask = data["view_mask"].astype(bool)
    tab = data["tabular"].astype(np.float32)
    has_tab = data["has_tabular"].astype(bool)
    t1_y = data["tier1_y"].astype(np.int64)
    t2_y = data["tier2_y"].astype(np.int64)
    can_y = data["canonical_y"].astype(np.int64)
    split = data["split"]

    embed_dim = emb.shape[-1]
    tabular_dim = tab.shape[-1]
    num_t1 = len(manifest["tier1_classes"])
    num_t2 = len(manifest["tier2_classes"])
    num_can = len(manifest["canonical_classes"])

    def make_loader(sp_code: int, shuffle: bool) -> DataLoader:
        idx = np.where(split == sp_code)[0]
        ds = TensorDataset(
            torch.from_numpy(emb[idx]),
            torch.from_numpy(mask[idx]),
            torch.from_numpy(tab[idx]),
            torch.from_numpy(has_tab[idx]),
            torch.from_numpy(t1_y[idx]),
            torch.from_numpy(t2_y[idx]),
            torch.from_numpy(can_y[idx]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(0, shuffle=True)
    val_loader = make_loader(1, shuffle=False)
    test_loader = make_loader(2, shuffle=False)

    model = NidaanMultimodalNet(
        embed_dim=embed_dim,
        tabular_dim=tabular_dim,
        num_tier1=num_t1,
        num_tier2=num_t2,
        num_canonical=num_can,
        stage=stage,
    ).to(device)

    use_focal = stage in {"stage3_adapter_hierarchical_focal", "stage4_multimodal_fusion"}
    t1_w = compute_inverse_freq_weights(t1_y[split == 0], num_t1, power=0.45).to(device) if use_focal else None
    can_w = compute_inverse_freq_weights(can_y[split == 0], num_can, power=0.45).to(device) if use_focal else None
    criterion = JointHierarchicalLoss(
        use_focal=use_focal,
        tier1_weights=t1_w,
        canonical_weights=can_w,
        w_tier1=0.35 if use_focal else 0.40,
        w_tier2=0.25 if use_focal else 0.15,
        w_canonical=0.40 if use_focal else 0.45,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -1.0
    best_state = None
    t0 = time.time()

    for _ in range(1, epochs + 1):
        model.train()
        for b_emb, b_mask, b_tab, b_ht, b_t1, b_t2, b_can in train_loader:
            b_emb, b_mask, b_tab, b_ht = b_emb.to(device), b_mask.to(device), b_tab.to(device), b_ht.to(device)
            b_t1, b_t2, b_can = b_t1.to(device), b_t2.to(device), b_can.to(device)

            if model.use_mil and torch.rand(1).item() < 0.30:
                multi = b_mask.sum(dim=1) > 1
                if multi.any():
                    drop_mask = b_mask.clone()
                    drop_mask[multi, 1:] = False
                    b_mask = drop_mask

            optimizer.zero_grad()
            if model.use_fusion:
                b_ht = b_ht & (~(torch.rand(b_ht.shape, device=device) < 0.25))
            out = model(b_emb, b_mask, b_tab, has_tabular=b_ht)
            loss = criterion(out, b_t1, b_t2, b_can)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_metrics = evaluate_split(model, val_loader, device, num_t1, num_can)
        score = 0.5 * val_metrics["canonical_macro_f1"] + 0.5 * val_metrics["tier1_macro_f1"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_raw = evaluate_split(model, val_loader, device, num_t1, num_can)
    opt_temp = fit_temperature(val_raw["can_logits"], torch.from_numpy(val_raw["can_y"]))
    model.temperature.fill_(opt_temp)

    train_raw = evaluate_split(model, train_loader, device, num_t1, num_can)
    val_cal = evaluate_split(model, val_loader, device, num_t1, num_can)
    ood_detector = MahalanobisOODDetector.fit(
        train_raw["reps"], train_raw["can_y"], val_cal["reps"], percentile=95.0
    )

    test_metrics = evaluate_split(model, test_loader, device, num_t1, num_can)
    _, test_ood_flags = ood_detector.score(test_metrics["reps"])

    summary = {
        "stage": stage,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_sec": round(time.time() - t0, 2),
        "calibrated_temperature": round(opt_temp, 4),
        "ood_threshold_95": round(ood_detector.threshold, 3),
        "test_ood_flag_rate": round(float(np.mean(test_ood_flags)), 4),
        "tier1_acc": test_metrics["tier1_acc"],
        "tier1_macro_f1": test_metrics["tier1_macro_f1"],
        "canonical_top1_acc": test_metrics["canonical_top1_acc"],
        "canonical_top3_acc": test_metrics["canonical_top3_acc"],
        "canonical_macro_f1": test_metrics["canonical_macro_f1"],
        "canonical_ece": test_metrics["canonical_ece"],
        "_ood_state": ood_detector.state_dict(),
    }
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Nidaan-Rx v2.0 Phase 3 4-Stage Ablation Trainer")
    parser.add_argument("--emb-dir", type=Path, default=Path("data/processed/v2_embeddings"))
    parser.add_argument("--backbone", default="panderm_base", choices=["panderm_base", "dinov2_small", "dinov2_base"])
    parser.add_argument("--out-dir", type=Path, default=Path("backend/artifacts"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = args.emb_dir / "dataset_manifest.json"
    npz_path = args.emb_dir / f"embeddings_{args.backbone}.npz"
    if not manifest_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path} or {npz_path}.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as raw:
        data = {k: raw[k] for k in raw.files}

    device = resolve_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Running 4-Stage Ablation on '{args.backbone}' | device={device} | cases={len(data['split']):,}")

    ablation_rows: list[dict[str, Any]] = []
    for st in STAGES:
        model, summary = train_single_stage(
            st, data, manifest, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed
        )
        ood_state = summary.pop("_ood_state")
        ablation_rows.append(summary)

        if st == "stage4_multimodal_fusion":
            ckpt_path = args.out_dir / f"nidaan_v2_{args.backbone}_stage4.pt"
            torch.save(
                {
                    "backbone": args.backbone,
                    "stage": st,
                    "embed_dim": int(data["embeddings"].shape[-1]),
                    "tabular_dim": int(data["tabular"].shape[-1]),
                    "tier1_classes": manifest["tier1_classes"],
                    "tier2_classes": manifest["tier2_classes"],
                    "canonical_classes": manifest["canonical_classes"],
                    "model_state_dict": model.cpu().state_dict(),
                    "ood_detector": ood_state,
                    "test_metrics": summary,
                },
                ckpt_path,
            )
            print(f"  -> Saved production Stage-4 checkpoint: {ckpt_path}")

    report_path = args.out_dir / f"ablation_report_{args.backbone}.json"
    report_path.write_text(json.dumps({"backbone": args.backbone, "stages": ablation_rows}, indent=2), encoding="utf-8")

    print(f"\n================ 4-STAGE ABLATION RESULTS ({args.backbone.upper()}) ================")
    print(f"{'Stage':<35} {'T1-Acc':>7} {'T1-F1':>7} {'Top1':>7} {'Top3':>7} {'Can-F1':>7} {'ECE':>7} {'Sec':>6}")
    print("-" * 92)
    for r in ablation_rows:
        print(
            f"{r['stage']:<35} {r['tier1_acc']*100:>6.2f}% {r['tier1_macro_f1']*100:>6.2f}% "
            f"{r['canonical_top1_acc']*100:>6.2f}% {r['canonical_top3_acc']*100:>6.2f}% "
            f"{r['canonical_macro_f1']*100:>6.2f}% {r['canonical_ece']:>7.4f} {r['train_sec']:>5.1f}s"
        )
    print(f"\nSaved full ablation JSON -> {report_path}")


if __name__ == "__main__":
    main()
