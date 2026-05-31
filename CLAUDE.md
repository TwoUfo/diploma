# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

University diploma project: **multitask deep learning on the NASA JPL Small-Body Database**. A shared-backbone MLP jointly predicts four properties of an asteroid from its orbital + observational metadata: (1) taxonomic **class**, (2) **diameter**, (3) geometric **albedo**, (4) **rotation period**. Source data (`data/raw/dataset.csv`, ~456 MB) is gitignored; processed parquet files and model checkpoints are generated locally and also gitignored.

> Absolute magnitude **H is no longer a predicted target** — it is a *model input feature* (the catalogue reports it for essentially every object; `preprocess` drops the ~0.7% of rows where it is missing). There is no `h_target`/`h_mask` in the processed schema, no H head, and no H term in the loss. The Streamlit app takes H as a slider and uses it in a closed-form diameter estimate. Don't reintroduce an H head expecting it to be wired up.

## Common commands

```bash
# Environment
source .venv/bin/activate
pip install -r requirements.txt

# Build processed splits (writes data/processed/*.parquet + scaler/encoder + class_albedo_prior)
python -m src.data.preprocessing

# Fetch the full raw catalogue from JPL (regenerates data/raw/dataset.csv)
python -m src.data.fetch_jpl_full

# Notebook-driven workflow (numbered, run in order on first setup)
jupyter notebook notebooks/

# Demo app (expects models/mtl_model.pt + data/processed/ + data/raw/dataset.csv to exist;
# the app uses relative paths, so run from inside app/)
cd app && streamlit run streamlit_app.py

# Container build (bundles processed data + trained model into the image)
docker build -t asteroid-mtl . && docker run -p 8501:8501 asteroid-mtl
```

There is no test runner wired up yet — [tests/](tests/) is empty. There is no lint config.

## Architecture

### The notebooks are the runnable pipeline

The `src/` modules export the reusable building blocks (`AsteroidMTLModel`, `MultiTaskLoss`, `AsteroidDataset`, `Trainer`, preprocessing helpers). The end-to-end wiring — computing class weights, constructing the optimizer/scheduler, the actual fit loop invocation, evaluation — lives in the **numbered notebooks**, and each depends on the previous one's on-disk artifacts:

1. `01_eda.ipynb` — profiling only, no outputs.
2. `02_preprocessing.ipynb` — produces `data/processed/{train,val,test}.parquet`, `scaler.joblib`, `label_encoder.joblib`, and the class-conditional albedo prior consumed by the model.
3. `03_baselines.ipynb` — LightGBM single-task baselines (the reference point for "does MTL help").
4. `04_mtl_training.ipynb` — produces `models/mtl_model.pt`. Computes inverse-frequency class weights here (not in the trainer).
5. `05_evaluation.ipynb` — MTL vs baselines, t-SNE on `shared_repr`, confusion matrix.

Preprocessing itself is fully runnable headless: `python -m src.data.preprocessing` calls `build_splits`, which writes the three parquet splits, `scaler.joblib`, `label_encoder.joblib`, and `class_albedo_prior.joblib`. The notebooks orchestrate the training/eval that consumes those artifacts.

All notebooks `sys.path.insert(0, '..')` and use `../data/...` / `../models/...` relative paths; the Streamlit app does the same. Keep this convention when adding scripts under those folders, or fix the paths globally.

### Data pipeline ([src/data/preprocessing.py](src/data/preprocessing.py))

The single source of truth for what becomes a feature and how targets are transformed. Non-obvious decisions that anything downstream must respect:

- **Target transforms** (see `preprocess`): diameter is trained in `log1p` space; albedo and rotation period in natural `log` space (both clipped before the log: albedo at `1e-4`, rotation at `0.01` h); class is label-encoded. The Streamlit app inverts these (`np.expm1` for diameter, `np.exp` for albedo/rotation).
- **Masked regression targets**: diameter, albedo, and rotation all have heavy missingness (rotation only ~2% coverage). Missing values are filled with 0 and accompanied by `diameter_mask` / `albedo_mask` / `rot_mask` columns (1 = observed, 0 = imputed). **Every regression loss and metric must apply these masks** — see [src/models/losses.py](src/models/losses.py), `regression_metrics` in [src/training/metrics.py](src/training/metrics.py), and the masked MDN NLL in [src/models/mtl_model.py](src/models/mtl_model.py).
- **Leak prevention / feature selection**: `DROP_COLUMNS` is a denylist (not an allowlist) covering identifiers (`id`, `spkid`, `pdes`, …), **leak columns** computed from / perfectly correlated with the targets (`q`, `ad`, `n`, `per`, `per_y`, `moid_ld`, `diameter_sigma`, and `albedo` — which is itself a target now), near-zero-coverage columns (`G`, `BV`, `UB`, `IR`, `spec_B`, `spec_T`), orbit-orientation-at-epoch columns with negligible signal (`om`, `w`, `tp`, `epoch*`), and `H_sigma` (only useful back when H was a target). After dropping those plus the class/diameter/rot targets, **every remaining numeric column becomes a feature** (`numeric_features = features.select_dtypes(...)`). Don't re-add a `DROP_COLUMNS` entry without checking why it's there.
- **Scaler is a `QuantileTransformer(output_distribution="normal")`**, fit on train only — not a plain StandardScaler. The Streamlit app relies on `scaler.feature_names_in_` to map inputs to the right columns.
- **Class-conditional albedo prior** (`class_albedo_prior.joblib`): per-class **mean** log-albedo over observed-albedo rows, computed from **train only**, saved as an artifact and loaded into the model as a buffer (see below). Index = encoded class id.
- Rare classes (`HYA`, `IEO`, `AST`, `CEN`) are folded into a single `Rare` label.
- Tisserand parameter w.r.t. Jupiter is engineered in `compute_tisserand` (`JUPITER_A = 5.2044`) and is one of the model's input features. Other Tisserand/resonance features were audited out as affine functions of `a`.

Processed parquet schema (target/mask columns enumerated in `AsteroidDataset` in [src/data/dataset.py](src/data/dataset.py)): feature columns (including H) + `class_label`, `diameter_target`, `albedo_target`, `rot_target`, `diameter_mask`, `albedo_mask`, `rot_mask`. `AsteroidDataset` separates features from these by name — **preserve the column names exactly** when changing preprocessing.

### Model ([src/models/mtl_model.py](src/models/mtl_model.py))

`AsteroidMTLModel` = a `SharedBackbone` (Linear → BN → ReLU → Dropout stack) feeding four heads. `forward` also returns `shared_repr` for representation analysis (t-SNE etc.) — don't remove it.

- **class** → `TaskHead` → logits.
- **diameter** → `TaskHead` → scalar (log1p space).
- **albedo** → `albedo_residual_head` predicts a *residual* that is added to a **class-conditional log-albedo prior** (`class_albedo_prior`, a registered buffer). `forward(x, class_label)` uses the prior of the *true* class when `class_label` is given (training); otherwise it uses the softmax-weighted average over class priors (inference). This is why the trainer passes the label during train and `None` during eval.
- **rotation** → `MDNHead`: a Mixture Density Network with `K=5` Gaussian components, modelling the bimodal YORP-spun vs primordial spin distribution instead of regressing to the population mean. `forward` returns `rot_log_pi`, `rot_mu`, `rot_log_sigma`; the point prediction `rot_pred` comes from `mdn_map` (mean of the most-likely component). Train it with `mdn_nll` (masked).

### Loss ([src/models/losses.py](src/models/losses.py))

`MultiTaskLoss` combines the four task losses with **Kendall et al. uncertainty weighting**: one learned `log_sigma_*` parameter per task (class, diam, albedo, rot). Classification uses weighted cross-entropy; diameter and albedo use **masked MSE**; rotation uses the **masked MDN NLL**. The `log_sigma_*` are `nn.Parameter`s on the criterion, so the optimizer must receive `criterion.parameters()` alongside the model's, and grad clipping spans both — see [src/training/trainer.py](src/training/trainer.py) (`list(self.model.parameters()) + list(self.criterion.parameters())`). If you swap the loss or optimizer setup, keep this wiring.

### Training loop ([src/training/trainer.py](src/training/trainer.py))

Generic train/validate loop with early stopping on val loss and grad clipping. `Trainer.__init__` accepts a `scheduler` argument and `fit` calls `scheduler.step()` each epoch when one is passed, but **nothing constructs a scheduler yet** (the YAML mentions `cosine`); build and pass it at the call site in notebook 04. Checkpoints (`save_checkpoint`) store `model`, `criterion`, and `optimizer` state dicts together — load all three so the learned `log_sigma_*` come back. The trainer calls `model(features, class_label)` during training (for the albedo prior) and `model(features, None)` during validation.

### Config ([src/utils/config.py](src/utils/config.py), [configs/default.yaml](configs/default.yaml))

`config.py` defines dataclasses mirroring `default.yaml`, but **nothing currently loads it** — notebooks and `preprocessing.py` hardcode their own constants (which match the YAML by convention). There are also name mismatches to reconcile before relying on `Config.from_yaml`: `ModelConfig` exposes `dropout_shared`/`dropout_head` and has no `backbone_dropouts`, `n_features`, `class_albedo_prior`, or `rot_mdn_components`, whereas `AsteroidMTLModel.__init__` expects `backbone_dropouts`/`head_dropout` and those extra args. The YAML's `drop_columns` list is also stale relative to the real `DROP_COLUMNS` in `preprocessing.py` — the code is the source of truth.

### Serving ([app/streamlit_app.py](app/streamlit_app.py))

Loads the checkpoint + `scaler`/`label_encoder`/`class_albedo_prior` + the train split at startup. **Artifact paths come from the `artifacts:` block in [configs/default.yaml](configs/default.yaml)** (resolved against the project root, so the app no longer depends on being run from `app/` for those) — don't hardcode them. Three input modes (sidebar radio):

- **Preset (real catalogued asteroid)**: a curated set of well-known objects (`PRESET_ASTEROID_PDES` in [src/data/build_presets.py](src/data/build_presets.py), e.g. Ceres, Vesta, Eros, Apophis, Pluto). Their *actual* full scaled feature vector is recovered by `compute_presets`. **Preset data has a CSV-or-artifact fallback**: when the ~465 MB `data/raw/dataset.csv` is present the app recomputes presets from it (freshest); otherwise it loads the small committed `data/processed/presets.joblib` (built once via `python -m src.data.build_presets`). A fresh clone has no CSV, so it uses the artifact. If you change preset feature engineering, rebuild that artifact.
- **Custom (k-NN inferred)**: the user sets only `e, a, i, ma, H`; `tisserand_j` is derived; the remaining features are filled via **distance-weighted k-NN** (`predict_with_knn_fill`) over the scaled orbital subspace of the train set. For this off-distribution path the MDN rotation output is *not* trusted — rotation is reported as the **global train median + IQR** instead.
- **All features (manual)**: the user types all 24 raw (pre-scaler) features directly; `neo`/`pha` render as checkboxes, `condition_code` as an int 0–9, the rest as floats. Each field has an **n/a checkbox** that substitutes the training median (`feat_median`). Rotation MDN *is* trusted here (in-distribution full vector).

The app reports **diameter two ways**: a physics closed-form `D[km] = 1329 / sqrt(p_v) · 10^(-H/5)` from the input H and predicted albedo, alongside the learned diameter head's value. Inputs map to the feature vector via `scaler.feature_names_in_`; keep the named-features path working (om/w were pruned in preprocessing as orientation-only, low-signal features).

### What's committed vs. gitignored

The ~465 MB raw CSV and the val/test parquet splits stay gitignored (regenerated via preprocessing). **Committed so a fresh clone runs the demo out of the box**: `models/mtl_model.pt` (~2.4 MB), the three demo artifacts `data/processed/{scaler,label_encoder,class_albedo_prior}.joblib`, the presets fallback `data/processed/presets.joblib`, and `data/processed/train.parquet` (~42 MB, needed for the k-NN mode). The `.gitignore` uses explicit `!`-negations for these — don't blanket-ignore `models/*.pt` or `data/processed/*.parquet` without re-adding them.

### Other directories

- `src/data/fetch_jpl_full.py` — pulls the full Small-Body Database to regenerate `data/raw/dataset.csv`.
- `src/data/build_presets.py` — `compute_presets` (shared by the app) + a `__main__` that pickles the small `presets.joblib` fallback. Run `python -m src.data.build_presets` after changing preset asteroids or preset feature engineering.
- `thesis_notes/` — markdown research notes feeding the written thesis (k-NN neighbor similarity study, diameter/albedo formula asymmetry, diameter outlier underestimation, parameter computation methods, glossary).
- `figures/` — generated plots referenced by the notes/thesis (k-NN containment, head-vs-formula diameter comparison).
