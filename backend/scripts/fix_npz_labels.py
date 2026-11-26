#!/usr/bin/env python3
"""
Local 2-second patch for Phase 2B cached embeddings (.npz + dataset_manifest.json).

Fixes 3 underscore-delimited DermaCon-IN labels without re-running Colab:
  1. seborrheic_dermatitis  -> Tier-1 scalp_follicular (6) / canonical seborrheic_dermatitis_scalp
  2. contact_dermatitis     -> Tier-1 eczematous_allergic (3) / canonical contact_dermatitis
  3. steroid_modified_tinea -> Tier-1 fungal_infections (1) / canonical tinea_incognito
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EMB_DIR = Path("data/processed/v2_embeddings")
MANIFEST_PATH = EMB_DIR / "dataset_manifest.json"
SEED = 42

OVERRIDES: dict[str, tuple[int, str]] = {
    "seborrheic_dermatitis": (6, "seborrheic_dermatitis_scalp"),
    "contact_dermatitis": (3, "contact_dermatitis"),
    "steroid_modified_tinea": (1, "tinea_incognito"),
}


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tier1_classes: list[str] = manifest["tier1_classes"]
    tier2_classes: list[str] = manifest["tier2_classes"]
    new_t2_to_t1 = dict(manifest["tier2_to_tier1"])
    new_t2_to_can = dict(manifest["tier2_to_canonical"])

    for t2_name, (t1_idx, can_name) in OVERRIDES.items():
        if t2_name in tier2_classes:
            new_t2_to_t1[t2_name] = t1_idx
            new_t2_to_can[t2_name] = can_name

    backbones = manifest["backbones_extracted"]
    with np.load(EMB_DIR / f"embeddings_{backbones[0]}.npz", allow_pickle=False) as raw:
        tier2_y = raw["tier2_y"]
        sources = raw["sources"]

    n = len(tier2_y)
    new_tier1_y = np.empty(n, dtype=np.int64)
    new_can_names: list[str] = []
    changed = Counter()

    for i, t2_idx in enumerate(tier2_y.tolist()):
        t2_name = tier2_classes[t2_idx]
        new_tier1_y[i] = new_t2_to_t1[t2_name]
        new_can_names.append(new_t2_to_can[t2_name])
        if t2_name in OVERRIDES:
            changed[t2_name] += 1

    canonical_vocab = sorted(set(new_can_names))
    can_to_idx = {c: i for i, c in enumerate(canonical_vocab)}
    new_can_y = np.array([can_to_idx[c] for c in new_can_names], dtype=np.int64)

    # Identical stratified 70/15/15 split by (source, canonical_y) across both backbones
    rng = np.random.default_rng(SEED)
    by_strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i in range(n):
        by_strata[(str(sources[i]), int(new_can_y[i]))].append(i)

    new_split = np.zeros(n, dtype=np.int8)
    for idxs in by_strata.values():
        arr = np.array(idxs, dtype=np.int64)
        rng.shuffle(arr)
        m = len(arr)
        n_test = max(1, int(round(0.15 * m))) if m >= 5 else 0
        n_val = max(1, int(round(0.15 * m))) if m >= 4 else 0
        new_split[arr[:n_test]] = 2
        new_split[arr[n_test : n_test + n_val]] = 1

    for bb in backbones:
        npz_path = EMB_DIR / f"embeddings_{bb}.npz"
        with np.load(npz_path, allow_pickle=False) as raw:
            arrays = {k: raw[k] for k in raw.files}
        arrays["tier1_y"] = new_tier1_y
        arrays["canonical_y"] = new_can_y
        arrays["split"] = new_split
        np.savez_compressed(npz_path, **arrays)
        print(f"Patched {npz_path.name}")

    split_names = {0: "train", 1: "val", 2: "test"}
    manifest["canonical_classes"] = canonical_vocab
    manifest["tier2_to_tier1"] = new_t2_to_t1
    manifest["tier2_to_canonical"] = new_t2_to_can
    manifest["tier1_counts"] = {
        tier1_classes[k]: int(v) for k, v in sorted(Counter(new_tier1_y.tolist()).items())
    }
    manifest["canonical_counts"] = {
        canonical_vocab[k]: int(v) for k, v in sorted(Counter(new_can_y.tolist()).items())
    }
    manifest["splits"] = {split_names[k]: int(v) for k, v in Counter(new_split.tolist()).items()}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nFixed underscore-mapped cases:", dict(changed))
    print("Updated Tier-1 counts:", manifest["tier1_counts"])
    print("Updated Canonical counts:", manifest["canonical_counts"])
    print("Updated Splits:", manifest["splits"])


if __name__ == "__main__":
    main()