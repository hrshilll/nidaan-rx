"""
Nidaan-Rx v2.0 — Publication-Ready Evaluation, Fairness, Calibration & OCR Figures.
Exports 6 high-DPI (300 DPI) figures to reports/figures/ and a Markdown + LaTeX
summary to reports/EVALUATION_SUMMARY.md using real held-out test artifacts.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.knowledge_graph import CONDITION_GRAPH, MedicineCatalog
from app.core.safety_gates import match_pmbjp_with_equivalence_gates
from app.models.fusion_net import NidaanMultimodalNet

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
EMB_DIR = PROJECT_ROOT / "data" / "processed" / "v2_embeddings"
RX_MANIFEST = PROJECT_ROOT / "synthetic_rx_dataset" / "manifests" / "ocr_baseline_test_all_750.json"

# Clinical Publication Palette
TEAL = "#0891B2"
DEEP_TEAL = "#164E63"
EMERALD = "#059669"
AMBER = "#D97706"
CORAL = "#DC2626"
SLATE = "#475569"
LIGHT_BG = "#F8FAFC"
GRID_COL = "#E2E8F0"

SHORT_CLASS_NAMES = {
    "acne_vulgaris": "Acne Vulgaris",
    "atopic_dermatitis_eczema": "Atopic Eczema",
    "bacterial_folliculitis_impetigo": "Folliculitis/Imp.",
    "benign_growth_nevus": "Benign Nevus",
    "contact_dermatitis": "Contact Derm.",
    "cutaneous_candidiasis": "Candidiasis",
    "herpes_viral_eruption": "Herpes Viral",
    "melasma_pih": "Melasma / PIH",
    "psoriasis_vulgaris": "Psoriasis",
    "scabies": "Scabies",
    "seborrheic_dermatitis_scalp": "Seborrheic Derm.",
    "suspected_neoplastic_lesion": "Neoplastic (Red)",
    "tinea_corporis_cruris": "Tinea Corporis",
    "tinea_incognito": "Tinea Incognito",
    "urticaria": "Urticaria",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.8,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> tuple[float, dict[str, np.ndarray]]:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts, bin_centers = [], [], [], []
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs >= lo) & (confs <= hi) if i == 0 else (confs > lo) & (confs <= hi)
        cnt = int(mask.sum())
        bin_counts.append(cnt)
        bin_centers.append((lo + hi) / 2.0)
        if cnt > 0:
            ba = float(accs[mask].mean())
            bc = float(confs[mask].mean())
            bin_accs.append(ba)
            bin_confs.append(bc)
            ece += (cnt / max(n, 1)) * abs(ba - bc)
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
    return float(ece), {
        "centers": np.array(bin_centers),
        "accs": np.array(bin_accs),
        "confs": np.array(bin_confs),
        "counts": np.array(bin_counts),
        "edges": edges,
    }


def compute_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_topk_acc(probs: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(probs, axis=1)[:, -k:]
    hits = np.any(topk == y_true[:, None], axis=1)
    return float(np.mean(hits))


def build_model_from_ckpt(ckpt: dict[str, Any]) -> NidaanMultimodalNet:
    sig = inspect.signature(NidaanMultimodalNet.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    kwargs: dict[str, Any] = {}
    cand = {
        "embed_dim": int(ckpt["embed_dim"]),
        "tabular_dim": int(ckpt["tabular_dim"]),
        "num_tier1": len(ckpt["tier1_classes"]),
        "num_tier2": len(ckpt["tier2_classes"]),
        "num_canonical": len(ckpt["canonical_classes"]),
        "stage": ckpt.get("stage", "stage4_multimodal_fusion"),
        "use_mil": True,
        "use_adapter": True,
        "use_fusion": True,
    }
    for k, v in cand.items():
        if k in params:
            kwargs[k] = v
    model = NidaanMultimodalNet(**kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def extract_canonical_logits(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        for key in ("canonical_logits", "logits_canonical", "canonical"):
            if key in out:
                return out[key]
        raise KeyError(f"Canonical logits not found in keys: {list(out.keys())}")
    if isinstance(out, (tuple, list)):
        return out[2]
    raise TypeError(f"Unexpected forward output type: {type(out)}")


def evaluate_backbone_on_test(backbone: str) -> dict[str, Any]:
    ckpt_path = ARTIFACTS_DIR / f"nidaan_v2_{backbone}_stage4.pt"
    npz_path = EMB_DIR / f"embeddings_{backbone}.npz"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt)

    npz = np.load(npz_path, allow_pickle=True)
    test_mask = npz["split"] == 2

    emb_t = torch.tensor(npz["embeddings"][test_mask].astype(np.float32))
    vmask_t = torch.tensor(npz["view_mask"][test_mask], dtype=torch.bool)
    tab_t = torch.tensor(npz["tabular"][test_mask].astype(np.float32))
    hastab_t = torch.tensor(npz["has_tabular"][test_mask], dtype=torch.bool)

    y_can = npz["canonical_y"][test_mask].astype(np.int64)
    y_t1 = npz["tier1_y"][test_mask].astype(np.int64)
    sources = npz["sources"][test_mask].astype(str)
    view_counts = npz["view_counts"][test_mask].astype(int)
    tab_np = npz["tabular"][test_mask].astype(np.float32)
    has_tab_np = npz["has_tabular"][test_mask].astype(bool)

    orig_temp = float(model.temperature.item())
    with torch.no_grad():
        # Calibrated (temperature-scaled)
        out_cal = model(emb_t, vmask_t, tab_t, has_tabular=hastab_t)
        logits_cal = extract_canonical_logits(out_cal)
        probs_cal = torch.softmax(logits_cal, dim=-1).numpy()

        # Uncalibrated (T = 1.0)
        model.temperature.fill_(1.0)
        out_raw = model(emb_t, vmask_t, tab_t, has_tabular=hastab_t)
        logits_raw = extract_canonical_logits(out_raw)
        probs_raw = torch.softmax(logits_raw, dim=-1).numpy()
        model.temperature.fill_(orig_temp)

        # Compute Mahalanobis OOD distances on test representations
        rep, _ = model.encode_representation(emb_t, vmask_t, tab_t, has_tabular=hastab_t)
        rep_np = rep.numpy()

    ood_scores = None
    ood_threshold = None
    ood_det = ckpt.get("ood_detector")
    if isinstance(ood_det, dict):
        ood_threshold = float(ood_det.get("threshold", ood_det.get("threshold_95", 24.832)))
        class_means = ood_det.get("class_means")
        precision = ood_det.get("precision")
        if class_means is not None and precision is not None:
            cm_vals = list(class_means.values()) if isinstance(class_means, dict) else list(class_means)
            means_arr = np.asarray(cm_vals, dtype=np.float64)
            prec_arr = np.asarray(precision, dtype=np.float64)
            dists = []
            for r in rep_np.astype(np.float64):
                diffs = means_arr - r[None, :]
                md2 = np.einsum("ij,jk,ik->i", diffs, prec_arr, diffs)
                dists.append(float(np.sqrt(max(float(np.min(md2)), 0.0))))
            ood_scores = np.array(dists, dtype=np.float64)

    return {
        "backbone": backbone,
        "temperature": orig_temp,
        "canonical_classes": list(ckpt["canonical_classes"]),
        "tier1_classes": list(ckpt["tier1_classes"]),
        "y_can": y_can,
        "y_t1": y_t1,
        "probs_cal": probs_cal,
        "probs_raw": probs_raw,
        "sources": sources,
        "view_counts": view_counts,
        "tabular": tab_np,
        "has_tabular": has_tab_np,
        "ood_scores": ood_scores,
        "ood_threshold": ood_threshold,
    }


# ==============================================================================
# FIGURE 1: 4-Stage Architectural Ablation (PanDerm-Base vs DINOv2-Small)
# ==============================================================================
def plot_fig1_ablation() -> dict[str, Any]:
    pan = json.loads((ARTIFACTS_DIR / "ablation_report_panderm_base.json").read_text())["stages"]
    dino = json.loads((ARTIFACTS_DIR / "ablation_report_dinov2_small.json").read_text())["stages"]

    stage_labels = ["S1\nLinear", "S2\nAttn-MIL", "S3\nAdapter+Focal", "S4\nMultimodal"]
    x = np.arange(len(stage_labels))

    panels = [
        ("tier1_acc", "Tier-1 Family Accuracy (%)", 100.0, (40, 88)),
        ("canonical_top1_acc", "15-Class Canonical Top-1 Accuracy (%)", 100.0, (35, 78)),
        ("canonical_top3_acc", "15-Class Canonical Top-3 Accuracy (%)", 100.0, (68, 96)),
        ("canonical_macro_f1", "15-Class Canonical Macro-F1 (%)", 100.0, (15, 72)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    fig.suptitle(
        "Nidaan-Rx v2.0 — 4-Stage Architectural Ablation (PanDerm-Base vs. DINOv2-Small, n = 1,057 Test)",
        fontsize=13,
        fontweight="bold",
        color=DEEP_TEAL,
        y=0.98,
    )

    for ax, (key, title, scale, ylim) in zip(axes.flat, panels):
        y_pan = [s[key] * scale for s in pan]
        y_dino = [s[key] * scale for s in dino]

        ax.plot(x, y_pan, marker="o", linewidth=2.4, markersize=7, color=TEAL, label="PanDerm-Base (768-d)")
        ax.plot(x, y_dino, marker="s", linewidth=2.2, markersize=6.5, linestyle="--", color=AMBER, label="DINOv2-Small (384-d)")

        for xi, yp, yd in zip(x, y_pan, y_dino):
            ax.annotate(f"{yp:.1f}%", (xi, yp), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.3, fontweight="bold", color=DEEP_TEAL)
            ax.annotate(f"{yd:.1f}%", (xi, yd), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.0, color="#92400E")

        ax.set_title(title, color=DEEP_TEAL, pad=8)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.grid(True, linestyle=":", color=GRID_COL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID_COL)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "fig1_ablation_backbone_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out.relative_to(PROJECT_ROOT)}")
    return {"panderm": pan, "dinov2": dino}


# ==============================================================================
# FIGURE 2: Reliability Diagram & Confidence Calibration (Before vs After Temp Scaling)
# ==============================================================================
def plot_fig2_calibration(pan_eval: dict[str, Any], dino_eval: dict[str, Any]) -> dict[str, Any]:
    ece_raw, bins_raw = compute_ece(pan_eval["probs_raw"], pan_eval["y_can"])
    ece_cal, bins_cal = compute_ece(pan_eval["probs_cal"], pan_eval["y_can"])
    ece_dino, bins_dino = compute_ece(dino_eval["probs_cal"], dino_eval["y_can"])

    brier_raw = compute_brier(pan_eval["probs_raw"], pan_eval["y_can"])
    brier_cal = compute_brier(pan_eval["probs_cal"], pan_eval["y_can"])
    brier_dino = compute_brier(dino_eval["probs_cal"], dino_eval["y_can"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.suptitle(
        "Probability Calibration & Reliability Audit (Held-Out Test Set, n = 1,057)",
        fontsize=12.5,
        fontweight="bold",
        color=DEEP_TEAL,
        y=0.99,
    )

    ax1.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.4, label="Perfect Calibration")

    mask_cal = ~np.isnan(bins_cal["accs"])
    mask_raw = ~np.isnan(bins_raw["accs"])
    mask_dino = ~np.isnan(bins_dino["accs"])

    ax1.bar(
        bins_cal["centers"][mask_cal],
        bins_cal["accs"][mask_cal],
        width=0.082,
        alpha=0.22,
        color=TEAL,
        edgecolor=TEAL,
        linewidth=1.1,
        label="PanDerm Calibrated Bins",
    )
    ax1.plot(
        bins_cal["confs"][mask_cal],
        bins_cal["accs"][mask_cal],
        marker="o",
        linewidth=2.3,
        color=DEEP_TEAL,
        label=f"PanDerm-Base Calibrated (T={pan_eval['temperature']:.2f}, ECE={ece_cal:.4f})",
    )
    ax1.plot(
        bins_raw["confs"][mask_raw],
        bins_raw["accs"][mask_raw],
        marker="^",
        linewidth=1.7,
        linestyle=":",
        color=CORAL,
        label=f"PanDerm-Base Uncalibrated (T=1.00, ECE={ece_raw:.4f})",
    )
    ax1.plot(
        bins_dino["confs"][mask_dino],
        bins_dino["accs"][mask_dino],
        marker="s",
        linewidth=1.8,
        linestyle="-.",
        color=AMBER,
        label=f"DINOv2-Small Calibrated (T={dino_eval['temperature']:.2f}, ECE={ece_dino:.4f})",
    )

    ax1.set_xlabel("Predicted Top-1 Confidence")
    ax1.set_ylabel("Empirical Top-1 Accuracy")
    ax1.set_title("Reliability Diagram (10 Equal-Width Bins)", color=DEEP_TEAL)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle=":", color=GRID_COL)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRID_COL, fontsize=7.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Confidence histogram split by Correct vs Incorrect predictions
    confs = pan_eval["probs_cal"].max(axis=1)
    correct = pan_eval["probs_cal"].argmax(axis=1) == pan_eval["y_can"]
    bins_h = np.linspace(0.0, 1.0, 16)
    ax2.hist(confs[correct], bins=bins_h, alpha=0.72, color=EMERALD, edgecolor="white", label=f"Correct ({int(correct.sum())})")
    ax2.hist(confs[~correct], bins=bins_h, alpha=0.65, color=CORAL, edgecolor="white", label=f"Misclassified ({int((~correct).sum())})")
    ax2.axvline(float(confs[correct].mean()), color="#047857", linestyle="--", linewidth=1.4, label=f"Mean Correct: {confs[correct].mean()*100:.1f}%")
    ax2.axvline(float(confs[~correct].mean()), color="#991B1B", linestyle="--", linewidth=1.4, label=f"Mean Wrong: {confs[~correct].mean()*100:.1f}%")
    ax2.set_xlabel("Calibrated Top-1 Confidence")
    ax2.set_ylabel("Test Case Count")
    ax2.set_title("Confidence Separation (PanDerm-Base Stage 4)", color=DEEP_TEAL)
    ax2.grid(True, linestyle=":", color=GRID_COL)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRID_COL, fontsize=7.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "fig2_calibration_reliability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out.relative_to(PROJECT_ROOT)}")
    return {
        "panderm_ece_raw": round(ece_raw, 4),
        "panderm_ece_cal": round(ece_cal, 4),
        "panderm_brier_raw": round(brier_raw, 4),
        "panderm_brier_cal": round(brier_cal, 4),
        "dinov2_ece_cal": round(ece_dino, 4),
        "dinov2_brier_cal": round(brier_dino, 4),
    }


# ==============================================================================
# FIGURE 3: 15-Class Row-Normalized Confusion Matrix (PanDerm-Base Stage 4)
# ==============================================================================
def plot_fig3_confusion_matrix(pan_eval: dict[str, Any]) -> None:
    classes = pan_eval["canonical_classes"]
    labels = [SHORT_CLASS_NAMES.get(c, c) for c in classes]
    y_true = pan_eval["y_can"]
    y_pred = pan_eval["probs_cal"].argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1)) * 100.0

    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    im = ax.imshow(cm_norm, cmap="YlGnBu", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label("Row-Normalized Recall (%)", fontsize=9.5, color=DEEP_TEAL)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=8.5)
    ax.set_yticklabels([f"{lbl} (n={int(row_sums[i, 0])})" for i, lbl in enumerate(labels)], fontsize=8.5)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            cnt = cm[i, j]
            if cnt > 0:
                txt_col = "white" if val >= 52 else "#0F172A"
                ax.text(j, i, f"{val:.0f}%\n({cnt})", ha="center", va="center", fontsize=6.9, color=txt_col, fontweight="bold" if i == j else "normal")

    ax.set_xlabel("Predicted Condition", fontsize=10.5, fontweight="bold", color=DEEP_TEAL, labelpad=8)
    ax.set_ylabel("Ground-Truth Condition", fontsize=10.5, fontweight="bold", color=DEEP_TEAL, labelpad=8)
    ax.set_title(
        "15-Class Canonical Confusion Matrix — PanDerm-Base Stage 4 (Test n = 1,057)",
        fontsize=12.5,
        fontweight="bold",
        color=DEEP_TEAL,
        pad=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig3_confusion_matrix_15class.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out.relative_to(PROJECT_ROOT)}")


# ==============================================================================
# FIGURE 4: Skin-Tone & Indian Cohort Subgroup Fairness + Per-Class F1/Top-3
# ==============================================================================
def plot_fig4_fairness_and_per_class(pan_eval: dict[str, Any]) -> dict[str, Any]:
    y_true = pan_eval["y_can"]
    probs = pan_eval["probs_cal"]
    y_pred = probs.argmax(axis=1)
    sources = pan_eval["sources"]
    tab = pan_eval["tabular"]
    has_tab = pan_eval["has_tabular"]

    # Reconstruct integer Monk Skin Tone (1..10) & Fitzpatrick (1..6) from normalized tabular columns [4] and [3]
    mst_vals = np.round(tab[:, 4] * 9.0 + 1.0).astype(int)
    fst_vals = np.round(tab[:, 3] * 5.0 + 1.0).astype(int)

    # Subgroup definitions: Indian Clinical Cohort (Dermacon-IN) + SCIN Skin Tone Strata
    scin_mask = (sources == "scin") & has_tab
    derm_mask = sources == "dermacon_in_hf"

    subgroups = [
        ("Dermacon-IN\n(Indian Clinical)", derm_mask),
        ("SCIN\nMonk 1-3 (Light)", scin_mask & (mst_vals <= 3)),
        ("SCIN\nMonk 4-6 (Med/IN)", scin_mask & (mst_vals >= 4) & (mst_vals <= 6)),
        ("SCIN\nMonk 7-10 (Deep)", scin_mask & (mst_vals >= 7)),
        ("Multi-View\n(2-3 Angles)", pan_eval["view_counts"] >= 2),
        ("Single-View\n(1 Angle)", pan_eval["view_counts"] == 1),
    ]

    subgroup_stats = []
    for name, m in subgroups:
        n = int(m.sum())
        if n == 0:
            continue
        top1 = float(accuracy_score(y_true[m], y_pred[m]))
        top3 = compute_topk_acc(probs[m], y_true[m], k=3)
        mf1 = float(f1_score(y_true[m], y_pred[m], average="macro", zero_division=0))
        ece_s, _ = compute_ece(probs[m], y_true[m], n_bins=8)
        subgroup_stats.append({
            "group": name.replace("\n", " "),
            "label": f"{name}\n(n={n})",
            "n": n,
            "top1": round(top1 * 100, 1),
            "top3": round(top3 * 100, 1),
            "macro_f1": round(mf1 * 100, 1),
            "ece": round(ece_s, 4),
        })

    # Per-class F1 & Top-3 recall
    classes = pan_eval["canonical_classes"]
    per_class = []
    for idx, cid in enumerate(classes):
        m = y_true == idx
        n_c = int(m.sum())
        if n_c == 0:
            continue
        c_f1 = float(f1_score((y_true == idx).astype(int), (y_pred == idx).astype(int), zero_division=0))
        c_top3 = compute_topk_acc(probs[m], y_true[m], k=3)
        per_class.append({
            "cid": cid,
            "name": SHORT_CLASS_NAMES.get(cid, cid),
            "n": n_c,
            "f1": round(c_f1 * 100, 1),
            "top3": round(c_top3 * 100, 1),
        })
    per_class.sort(key=lambda d: d["f1"], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.suptitle(
        "Subgroup Fairness Audit (Indian Cohort & Monk Skin Tone) + Per-Condition Performance",
        fontsize=12.5,
        fontweight="bold",
        color=DEEP_TEAL,
        y=0.99,
    )

    # Left: Subgroup grouped bars
    x = np.arange(len(subgroup_stats))
    w = 0.26
    t1_vals = [s["top1"] for s in subgroup_stats]
    t3_vals = [s["top3"] for s in subgroup_stats]
    f1_vals = [s["macro_f1"] for s in subgroup_stats]

    b1 = ax1.bar(x - w, t1_vals, width=w, color=TEAL, label="Top-1 Accuracy (%)")
    b2 = ax1.bar(x, t3_vals, width=w, color=EMERALD, label="Top-3 Accuracy (%)")
    b3 = ax1.bar(x + w, f1_vals, width=w, color=AMBER, label="Macro-F1 (%)")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#1E293B")

    ax1.set_xticks(x)
    ax1.set_xticklabels([s["label"] for s in subgroup_stats], fontsize=7.8)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Cohort, Skin-Tone & Multi-View Parity", color=DEEP_TEAL)
    ax1.grid(axis="y", linestyle=":", color=GRID_COL)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID_COL, fontsize=8.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Per-class horizontal bars
    y_pos = np.arange(len(per_class))
    ax2.barh(y_pos + 0.18, [c["top3"] for c in per_class], height=0.35, color="#A7F3D0", edgecolor=EMERALD, label="Top-3 Recall (%)")
    ax2.barh(y_pos - 0.18, [c["f1"] for c in per_class], height=0.35, color=TEAL, label="F1-Score (%)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{c['name']} (n={c['n']})" for c in per_class], fontsize=8.1)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 112)
    ax2.set_xlabel("Score (%)")
    ax2.set_title("Per-Condition F1 vs. Top-3 Recall (15 Classes)", color=DEEP_TEAL)
    ax2.grid(axis="x", linestyle=":", color=GRID_COL)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID_COL, fontsize=8.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for idx_c, c in enumerate(per_class):
        ax2.text(c["f1"] + 1.2, idx_c - 0.18, f"{c['f1']:.0f}%", va="center", fontsize=7.2, color=DEEP_TEAL, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "fig4_fairness_subgroup_audit.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out.relative_to(PROJECT_ROOT)}")
    return {"subgroups": subgroup_stats, "per_class": per_class}


# ==============================================================================
# FIGURE 5: Multilingual Indian Prescription OCR Benchmark (n = 750 Test)
# ==============================================================================
def plot_fig5_ocr_benchmark() -> dict[str, Any]:
    ocr = json.loads(RX_MANIFEST.read_text(encoding="utf-8"))
    by_tier = ocr["by_tier"]
    by_lang = ocr["by_language"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
    fig.suptitle(
        "Multilingual Prescription OCR Benchmark — PaddleOCR v3 (n = 750 Synthetic Indian Rx Test Set)",
        fontsize=12.5,
        fontweight="bold",
        color=DEEP_TEAL,
        y=0.99,
    )

    tiers = [
        ("tier1_clean", f"Tier 1: Clean\n(n={by_tier['tier1_clean']['rx_images']})"),
        ("tier2_smartphone", f"Tier 2: Smartphone\n(n={by_tier['tier2_smartphone']['rx_images']})"),
        ("tier3_hard", f"Tier 3: Hard Warp\n(n={by_tier['tier3_hard']['rx_images']})"),
    ]
    x = np.arange(len(tiers))
    w = 0.21
    m_char = [by_tier[k]["brand_salt_char_acc"] * 100 for k, _ in tiers]
    m_exact = [by_tier[k]["salt_recall_exact"] * 100 for k, _ in tiers]
    m_fuzzy = [by_tier[k]["salt_recall_fuzzy_85"] * 100 for k, _ in tiers]
    m_dose = [by_tier[k]["dosage_exact_acc"] * 100 for k, _ in tiers]

    for offset, vals, col, lbl in [
        (-1.5 * w, m_char, DEEP_TEAL, "Char Acc (%)"),
        (-0.5 * w, m_exact, TEAL, "Salt Exact Recall (%)"),
        (0.5 * w, m_fuzzy, EMERALD, "Salt Fuzzy-85 Recall (%)"),
        (1.5 * w, m_dose, AMBER, "Dosage Exact Acc (%)"),
    ]:
        bars = ax1.bar(x + offset, vals, width=w, color=col, label=lbl)
        for b in bars:
            h = b.get_height()
            ax1.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.0, fontweight="bold")

    ax1.axhline(95.0, linestyle="--", color=CORAL, linewidth=1.2, label="95% Clinical Gate")
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in tiers])
    ax1.set_ylim(88, 102.2)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Accuracy Across Imaging Difficulty Tiers", color=DEEP_TEAL)
    ax1.grid(axis="y", linestyle=":", color=GRID_COL)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRID_COL, fontsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Script / Language Robustness
    lang_map = [("en", "English"), ("en_gu", "English + Gujarati"), ("en_hi", "English + Hindi")]
    langs = [(k, f"{lbl}\n(n={by_lang[k]['rx_images']})") for k, lbl in lang_map if k in by_lang]
    xl = np.arange(len(langs))
    wl = 0.24
    l_char = [by_lang[k]["brand_salt_char_acc"] * 100 for k, _ in langs]
    l_fuzzy = [by_lang[k]["salt_recall_fuzzy_85"] * 100 for k, _ in langs]
    l_dose = [by_lang[k]["dosage_exact_acc"] * 100 for k, _ in langs]

    for offset, vals, col, lbl in [
        (-wl, l_char, DEEP_TEAL, "Brand/Salt Char Acc (%)"),
        (0.0, l_fuzzy, EMERALD, "Salt Fuzzy-85 Recall (%)"),
        (wl, l_dose, AMBER, "Dosage Exact Acc (%)"),
    ]:
        bars = ax2.bar(xl + offset, vals, width=wl, color=col, label=lbl)
        for b in bars:
            h = b.get_height()
            ax2.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.4, fontweight="bold")

    ax2.axhline(95.0, linestyle="--", color=CORAL, linewidth=1.2, label="95% Clinical Gate")
    ax2.set_xticks(xl)
    ax2.set_xticklabels([lbl for _, lbl in langs])
    ax2.set_ylim(88, 102.2)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Multilingual Robustness (Zero-Hallucination on Hard Neg: 100%)", color=DEEP_TEAL)
    ax2.grid(axis="y", linestyle=":", color=GRID_COL)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRID_COL, fontsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "fig5_ocr_benchmark_multilingual.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out.relative_to(PROJECT_ROOT)}")
    return ocr["overall"]


# ==============================================================================
# FIGURE 6: Mahalanobis OOD Uncertainty Gate + PMBJP Jan Aushadhi Savings
# ==============================================================================
def plot_fig6_ood_and_pmbjp(pan_eval: dict[str, Any]) -> dict[str, Any]:
    cat = MedicineCatalog(
        pmbjp_csv=PROJECT_ROOT / "data" / "processed" / "pmbjp_dermatology_parsed.csv",
        onemg_csv=PROJECT_ROOT / "data" / "processed" / "1mg_dermatology_parsed.csv",
    )

    savings_rows = []
    seen_opts = set()
    for ck in CONDITION_GRAPH.values():
        for opt in ck.otc_options + ck.rx_discussion_options:
            if opt.option_id in seen_opts or not opt.salts:
                continue
            seen_opts.add(opt.option_id)
            matches = match_pmbjp_with_equivalence_gates(opt, cat, [])
            if matches:
                m = matches[0]
                if m.mrp_inr and m.mrp_inr > 0 and m.commercial_avg_inr and m.commercial_avg_inr > m.mrp_inr:
                    short_lbl = opt.display_name.split("(")[0].strip()
                    if len(short_lbl) > 28:
                        short_lbl = short_lbl[:26] + ".."
                    sku_key = str(m.generic_code or m.product_name).strip().lower()
                    savings_rows.append({
                        "sku_key": sku_key,
                        "label": short_lbl,
                        "pmbjp_inr": float(m.mrp_inr),
                        "comm_inr": float(m.commercial_avg_inr),
                        "savings_pct": float(m.estimated_savings_pct or 0.0),
                    })

    # Deduplicate by unique PMBJP SKU so every bar is a distinct medication
    dedup: dict[str, dict[str, Any]] = {}
    for r in savings_rows:
        k = r["sku_key"]
        if k not in dedup or len(r["label"]) < len(dedup[k]["label"]):
            dedup[k] = r
    savings_rows = sorted(dedup.values(), key=lambda r: r["savings_pct"], reverse=True)[:12]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.suptitle(
        "Safety & Economic Impact — Gate 4 Mahalanobis OOD Gating & Jan Aushadhi (PMBJP) Cost Reduction",
        fontsize=12.2,
        fontweight="bold",
        color=DEEP_TEAL,
        y=0.99,
    )

    # Left: OOD distribution
    ood_scores = pan_eval["ood_scores"]
    thr = pan_eval["ood_threshold"] or 24.832
    if ood_scores is not None:
        in_dist = ood_scores <= thr
        ood_flag_rate = float(np.mean(~in_dist)) * 100.0
        bins = np.linspace(float(np.percentile(ood_scores, 0.5)), float(np.percentile(ood_scores, 99.5)), 28)
        ax1.hist(ood_scores[in_dist], bins=bins, color=TEAL, alpha=0.72, edgecolor="white", label=f"In-Distribution ({int(in_dist.sum())})")
        ax1.hist(ood_scores[~in_dist], bins=bins, color=CORAL, alpha=0.80, edgecolor="white", label=f"Flagged OOD ({int((~in_dist).sum())} | {ood_flag_rate:.1f}%)")
        ax1.axvline(thr, color="#991B1B", linestyle="--", linewidth=1.8, label=f"95% Gate Threshold ({thr:.2f})")
        ax1.set_xlabel("Minimum Class-Conditional Mahalanobis Distance")
        ax1.set_ylabel("Held-Out Test Cases")
        ax1.set_title("Gate 4 Mahalanobis OOD Distribution (PanDerm-Base)", color=DEEP_TEAL)
        ax1.grid(axis="y", linestyle=":", color=GRID_COL)
        ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID_COL, fontsize=8.0)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
    else:
        ax1.text(0.5, 0.5, "OOD Detector Matrix Not Available", ha="center", va="center")

    # Right: PMBJP vs Commercial MRP horizontal dumbbell / bar comparison
    y_p = np.arange(len(savings_rows))
    ax2.barh(y_p - 0.18, [r["comm_inr"] for r in savings_rows], height=0.34, color="#CBD5E1", edgecolor=SLATE, label="1mg Commercial Avg MRP (₹)")
    ax2.barh(y_p + 0.18, [r["pmbjp_inr"] for r in savings_rows], height=0.34, color=EMERALD, label="PMBJP Jan Aushadhi MRP (₹)")
    ax2.set_yticks(y_p)
    ax2.set_yticklabels([r["label"] for r in savings_rows], fontsize=8.3)
    ax2.invert_yaxis()
    max_x = max((r["comm_inr"] for r in savings_rows), default=200) * 1.28
    ax2.set_xlim(0, max_x)
    ax2.set_xlabel("Price per Pack / Tube (INR ₹)")
    ax2.set_title("Verified Gates 1–3 Dermatology Generics: Commercial vs. PMBJP", color=DEEP_TEAL)
    ax2.grid(axis="x", linestyle=":", color=GRID_COL)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID_COL, fontsize=8.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for i, r in enumerate(savings_rows):
        ax2.text(
            r["comm_inr"] + max_x * 0.015,
            i,
            f"₹{r['pmbjp_inr']:.0f} vs ₹{r['comm_inr']:.0f} (-{r['savings_pct']:.0f}%)",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#047857",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "fig6_pmbjp_savings_and_ood.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out.relative_to(PROJECT_ROOT)}")

    mean_sav = float(np.mean([r["savings_pct"] for r in savings_rows])) if savings_rows else 0.0
    med_sav = float(np.median([r["savings_pct"] for r in savings_rows])) if savings_rows else 0.0
    return {
        "matched_essentials": len(savings_rows),
        "mean_savings_pct": round(mean_sav, 1),
        "median_savings_pct": round(med_sav, 1),
        "rows": savings_rows,
    }


def write_evaluation_summary(
    ablation: dict[str, Any],
    calib: dict[str, Any],
    fairness: dict[str, Any],
    ocr_overall: dict[str, Any],
    pmbjp: dict[str, Any],
) -> None:
    pan_s4 = ablation["panderm"][-1]
    dino_s4 = ablation["dinov2"][-1]

    lines = [
        "# Nidaan-Rx v2.0 — Evaluation, Fairness & Benchmark Summary",
        "",
        "All figures are saved at **300 DPI** under `reports/figures/` and generated directly from held-out test checkpoints (`n = 1,057` clinical dermatology cases; `n = 750` synthetic multilingual prescriptions; 160 PMBJP & 13,432 1mg SKUs).",
        "",
        "## 1. Vision Backbone & 4-Stage Ablation (`fig1_ablation_backbone_comparison.png`)",
        "",
        "| Backbone | Stage | Tier-1 Acc | Canonical Top-1 | Canonical Top-3 | Macro-F1 | ECE | Params |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for b_name, stages in [("PanDerm-Base", ablation["panderm"]), ("DINOv2-Small", ablation["dinov2"])]:
        for s in stages:
            lines.append(
                f"| {b_name} | `{s['stage']}` | {s['tier1_acc']*100:.2f}% | {s['canonical_top1_acc']*100:.2f}% | "
                f"{s['canonical_top3_acc']*100:.2f}% | {s['canonical_macro_f1']*100:.2f}% | {s['canonical_ece']:.4f} | {s['trainable_params']:,} |"
            )

    lines += [
        "",
        "## 2. Calibration & Reliability (`fig2_calibration_reliability.png`)",
        "",
        f"- **PanDerm-Base Stage 4:** ECE improved from `{calib['panderm_ece_raw']:.4f}` (uncalibrated) to **`{calib['panderm_ece_cal']:.4f}`** (temperature-scaled); Brier Score = **`{calib['panderm_brier_cal']:.4f}`**.",
        f"- **DINOv2-Small Stage 4:** Calibrated ECE = **`{calib['dinov2_ece_cal']:.4f}`**; Brier Score = **`{calib['dinov2_brier_cal']:.4f}`**.",
        "",
        "## 3. Subgroup Fairness & Indian Cohort Parity (`fig4_fairness_subgroup_audit.png`)",
        "",
        "| Subgroup | n | Top-1 Acc | Top-3 Acc | Macro-F1 | ECE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for sg in fairness["subgroups"]:
        lines.append(f"| {sg['group']} | {sg['n']} | {sg['top1']:.1f}% | {sg['top3']:.1f}% | {sg['macro_f1']:.1f}% | {sg['ece']:.4f} |")

    lines += [
        "",
        "## 4. Multilingual Prescription OCR Benchmark (`fig5_ocr_benchmark_multilingual.png`)",
        "",
        f"- **Test Set:** `n = {ocr_overall['rx_images']}` prescriptions + `{ocr_overall['hn_images']}` hard negatives (lab reports / clinic bills).",
        f"- **Brand/Salt Character Accuracy:** **`{ocr_overall['brand_salt_char_acc']*100:.2f}%`** (CER = `{ocr_overall['brand_salt_cer']*100:.2f}%`).",
        f"- **Salt Recall (Fuzzy-85 / Exact):** **`{ocr_overall['salt_recall_fuzzy_85']*100:.2f}%`** / **`{ocr_overall['salt_recall_exact']*100:.2f}%`**.",
        f"- **Dosage Exact Accuracy:** **`{ocr_overall['dosage_exact_acc']*100:.2f}%`**.",
        f"- **Hard-Negative Zero-Hallucination Rate:** **`{ocr_overall['hn_zero_hallucination_rate']*100:.1f}%`** (0 false drug extractions).",
        "",
        "## 5. Jan Aushadhi (PMBJP) Economic Impact (`fig6_pmbjp_savings_and_ood.png`)",
        "",
        f"- Across **{pmbjp['matched_essentials']}** verified core dermatology medications (Gates 1–3 exact salt, strength & vehicle match), PMBJP Jan Aushadhi generics achieve a **mean cost reduction of {pmbjp['mean_savings_pct']:.1f}%** (**median {pmbjp['median_savings_pct']:.1f}%**).",
        "",
        "| Medication | 1mg Commercial Avg | PMBJP Jan Aushadhi MRP | Savings |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for r in pmbjp["rows"]:
        lines.append(f"| {r['label']} | ₹{r['comm_inr']:.0f} | ₹{r['pmbjp_inr']:.0f} | **{r['savings_pct']:.1f}%** |")

    lines += [
        "",
        "## 6. Copy-Paste LaTeX Summary Table",
        "",
        "```latex",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Nidaan-Rx v2.0 Stage-4 Held-Out Test Performance ($n=1{,}057$)}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Backbone} & \textbf{Tier-1 Acc} & \textbf{Top-1 Acc} & \textbf{Top-3 Acc} & \textbf{Macro-F1} & \textbf{ECE} \\",
        r"\hline",
        f"DINOv2-Small (384-d) & {dino_s4['tier1_acc']*100:.1f}\\% & {dino_s4['canonical_top1_acc']*100:.1f}\\% & {dino_s4['canonical_top3_acc']*100:.1f}\\% & {dino_s4['canonical_macro_f1']*100:.1f}\\% & {dino_s4['canonical_ece']:.4f} \\\\",
        f"PanDerm-Base (768-d) & \\textbf{{{pan_s4['tier1_acc']*100:.1f}\\%}} & \\textbf{{{pan_s4['canonical_top1_acc']*100:.1f}\\%}} & \\textbf{{{pan_s4['canonical_top3_acc']*100:.1f}\\%}} & \\textbf{{{pan_s4['canonical_macro_f1']*100:.1f}\\%}} & \\textbf{{{pan_s4['canonical_ece']:.4f}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "```",
    ]

    out_md = PROJECT_ROOT / "reports" / "EVALUATION_SUMMARY.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Saved {out_md.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("Running Nidaan-Rx v2.0 evaluation figure pipeline...")
    ablation = plot_fig1_ablation()
    pan_eval = evaluate_backbone_on_test("panderm_base")
    dino_eval = evaluate_backbone_on_test("dinov2_small")
    calib = plot_fig2_calibration(pan_eval, dino_eval)
    plot_fig3_confusion_matrix(pan_eval)
    fairness = plot_fig4_fairness_and_per_class(pan_eval)
    ocr_overall = plot_fig5_ocr_benchmark()
    pmbjp = plot_fig6_ood_and_pmbjp(pan_eval)
    write_evaluation_summary(ablation, calib, fairness, ocr_overall, pmbjp)
    print("\nAll 6 figures and EVALUATION_SUMMARY.md generated successfully!")


if __name__ == "__main__":
    main()
