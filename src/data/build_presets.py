"""Build a tiny *presets* artifact for the Streamlit demo.

The app shows a handful of real, catalogued asteroids (Ceres, Vesta, …) whose
true feature vectors are recovered by running the preprocessing pipeline over
the full raw catalogue (``data/raw/dataset.csv``, ~465 MB). That CSV is far too
large for GitHub, so a fresh clone won't have it.

This module factors out the catalogue-dependent computation (`compute_presets`)
so it can be used two ways:

* The Streamlit app calls `compute_presets` directly **when the raw CSV is
  present** — always the freshest source of truth.
* Run as a script (`python -m src.data.build_presets`) it pickles the result to
  the small ``presets_path`` artifact, which the app loads as a **fallback when
  the CSV is absent**.

Paths come from ``configs/default.yaml`` (the ``artifacts`` section).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Curated set of real, well-known asteroids spanning the taxonomic range.
PRESET_ASTEROID_PDES = [
    "1",        # Ceres — largest MBA / dwarf planet
    "4",        # Vesta
    "65",       # Cybele
    "433",      # Eros — first NEO discovered
    "434",      # Hungaria
    "588",      # Achilles — first Jupiter Trojan
    "1862",     # Apollo
    "2060",     # Chiron — first Centaur
    "2062",     # Aten
    "5261",     # Eureka — first Mars Trojan
    "99942",    # Apophis
    "134340",   # Pluto
]

RARE_REMAP = {"HYA": "Rare", "IEO": "Rare", "AST": "Rare", "CEN": "Rare"}

# Columns shown in the preset UI (orbital params + true physical values).
DISPLAY_COLS = ["pdes", "name", "class", "e", "a", "i", "om", "w", "ma",
                "H", "diameter", "albedo", "rot_per"]


def compute_presets(raw_path: str, scaler, feat_names: list[str],
                    preset_pdes: list[str] = PRESET_ASTEROID_PDES) -> dict:
    """Recover preset feature vectors + display rows + manual-entry stats.

    Everything in the returned dict depends only on the raw catalogue (plus the
    fitted ``scaler``), so it is exactly what the app needs when the CSV is gone.
    """
    from src.data.preprocessing import preprocess, drop_corrupt_rows

    full_raw = pd.read_csv(raw_path, low_memory=False)
    full_raw["pdes_str"] = full_raw["pdes"].astype(str).str.strip()
    full_clean = drop_corrupt_rows(full_raw)
    full_features, _, _, _, _ = preprocess(full_raw.copy())

    preset_set = set(preset_pdes)
    preset_clean_idx = full_clean.index[full_clean["pdes_str"].isin(preset_set)].tolist()
    aligned = full_features.loc[preset_clean_idx, feat_names]

    preset_scaled_matrix = scaler.transform(aligned.values).astype(np.float32)
    preset_raw_matrix = aligned.values.astype(np.float64)
    pdes_order = [full_clean.loc[idx, "pdes_str"] for idx in preset_clean_idx]
    preset_scaled = dict(zip(pdes_order, preset_scaled_matrix))
    preset_raw = dict(zip(pdes_order, preset_raw_matrix))

    # Display rows (true catalogue values), ordered like PRESET_ASTEROID_PDES.
    disp = full_raw[full_raw["pdes_str"].isin(preset_set)][DISPLAY_COLS + ["pdes_str"]].copy()
    disp["pdes"] = disp["pdes_str"].astype(str).str.strip()
    disp["taxonomy"] = disp["class"].replace(RARE_REMAP)
    preset_rows = disp.drop(columns="pdes_str").set_index("pdes")
    preset_rows = preset_rows.loc[[p for p in preset_pdes if p in preset_rows.index]]

    # Per-feature stats in the pre-scaler space — defaults/hints for manual entry.
    feat_median = full_features[feat_names].median().values.astype(np.float64)
    feat_min = full_features[feat_names].min().values.astype(np.float64)
    feat_max = full_features[feat_names].max().values.astype(np.float64)

    return {
        "feat_names": list(feat_names),
        "preset_scaled": preset_scaled,
        "preset_raw": preset_raw,
        "preset_rows": preset_rows,
        "feat_median": feat_median,
        "feat_min": feat_min,
        "feat_max": feat_max,
    }


def _load_paths() -> dict:
    import yaml
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["artifacts"]


def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    art = _load_paths()
    scaler = joblib.load(PROJECT_ROOT / art["scaler_path"])
    feat_names = list(scaler.feature_names_in_)

    raw_path = PROJECT_ROOT / art["raw_dataset_path"]
    if not raw_path.exists():
        raise SystemExit(
            f"Raw dataset not found at {raw_path} — cannot build presets. "
            f"Fetch it first with: python -m src.data.fetch_jpl_full"
        )

    presets = compute_presets(str(raw_path), scaler, feat_names)
    out_path = PROJECT_ROOT / art["presets_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(presets, out_path)

    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB) with {len(presets['preset_scaled'])} presets.")


if __name__ == "__main__":
    main()
