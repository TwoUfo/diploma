# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

University diploma project: **multitask deep learning on the NASA JPL Small-Body Database**. A shared-backbone MLP jointly predicts (1) asteroid taxonomic class, (2) absolute magnitude H, (3) diameter. Source data (`data/raw/dataset.csv`, ~456 MB) is gitignored; processed parquet files and model checkpoints are generated locally and also gitignored.

## Common commands

```bash
# Environment
source .venv/bin/activate
pip install -r requirements.txt

# Build processed splits (writes data/processed/*.parquet + scaler/encoder)
python -m src.data.preprocessing

# Notebook-driven workflow (numbered, run in order on first setup)
jupyter notebook notebooks/

# Demo app (expects models/best_mtl_model.pt and data/processed/ to exist;
# the app uses relative paths, so run from inside app/)
cd app && streamlit run streamlit_app.py

# Container build (bundles processed data + trained model into the image)
docker build -t asteroid-mtl . && docker run -p 8501:8501 asteroid-mtl
```

There is no test runner wired up yet — [tests/](tests/) is empty. There is no lint config.

## Architecture

### Data pipeline ([src/data/preprocessing.py](src/data/preprocessing.py))

The single source of truth for what becomes a feature. Two non-obvious decisions that anything downstream must respect:

- **Leak prevention**: `diameter_sigma`, `albedo`, and several derived orbital quantities (`q`, `ad`, `n`, `per`, `per_y`, `moid_ld`) are dropped before training because they are computed from or perfectly correlated with the regression targets. Do not re-add them without checking `DROP_COLUMNS`.
- **Masked regression targets**: H and diameter have heavy missingness. They are filled with 0 and accompanied by `h_mask` / `diameter_mask` columns (1 = observed, 0 = imputed). All regression losses and metrics downstream **must** apply these masks — see [src/models/losses.py](src/models/losses.py) and [src/training/metrics.py](src/training/metrics.py).
- Rare classes (`HYA`, `IEO`, `AST`, `CEN`) are folded into a single `Rare` label.
- Diameter is trained in `log1p` space; the Streamlit app calls `np.expm1` on the prediction.
- Tisserand parameter w.r.t. Jupiter is engineered in `compute_tisserand` and is one of the model's input features.

Processed parquet files have a fixed schema: feature columns + `class_label`, `h_target`, `diameter_target`, `h_mask`, `diameter_mask`. `AsteroidDataset` ([src/data/dataset.py](src/data/dataset.py)) splits these by name — preserve the column names exactly when changing preprocessing.

### Model & loss

- [src/models/mtl_model.py](src/models/mtl_model.py): `SharedBackbone` (Linear → BN → ReLU → Dropout stack) feeds three independent `TaskHead`s. `forward` also returns `shared_repr` for representation analysis (t-SNE etc.) — don't remove it.
- [src/models/losses.py](src/models/losses.py): `MultiTaskLoss` implements **Kendall et al. uncertainty weighting** with learned `log_sigma_*` parameters. These are `nn.Parameter`s on the criterion, so the trainer must pass `criterion.parameters()` to the optimizer alongside the model's — see [src/training/trainer.py:62-65](src/training/trainer.py#L62-L65). If you swap the loss or optimizer setup, keep this wiring.

### Training loop ([src/training/trainer.py](src/training/trainer.py))

Generic train/validate loop with early stopping on val loss and grad clipping. Notable gap: `self.scheduler.step()` is called every epoch, but no scheduler is constructed anywhere in the codebase yet — the YAML mentions `cosine` but it isn't wired up. Add the scheduler at the call site (notebook 04) or extend `Trainer.__init__`.

### Config

[src/utils/config.py](src/utils/config.py) defines dataclasses for [configs/default.yaml](configs/default.yaml), but **nothing currently loads it**. Notebooks and `preprocessing.py` hardcode their own constants (which match the YAML by convention). Also note a name mismatch: `ModelConfig` exposes `dropout_shared`/`dropout_head`, while `AsteroidMTLModel.__init__` expects `backbone_dropouts`/`head_dropout`. Reconcile before relying on `Config.from_yaml`.

### Notebooks are the entry points

Pipeline state lives in artifacts on disk, and each notebook depends on the previous one having been run:

1. `01_eda.ipynb` — profiling only, no outputs.
2. `02_preprocessing.ipynb` — produces `data/processed/{train,val,test}.parquet`, `scaler.joblib`, `label_encoder.joblib`.
3. `03_baselines.ipynb` — LightGBM single-task baselines (the reference point for "does MTL help").
4. `04_mtl_training.ipynb` — produces `models/best_mtl_model.pt`. Computes inverse-frequency class weights here, not in the trainer.
5. `05_evaluation.ipynb` — MTL vs baselines, t-SNE on `shared_repr`, confusion matrix.

All notebooks `sys.path.insert(0, '..')` and use `../data/...` / `../models/...` relative paths. The Streamlit app does the same. Keep this convention when adding new notebooks/scripts under those folders, or fix the paths globally.

### Serving ([app/streamlit_app.py](app/streamlit_app.py))

Loads `scaler`, `label_encoder`, and the checkpoint, then maps slider inputs to the feature vector via `scaler.feature_names_in_`. There is a hardcoded 22-zero fallback if `feature_names_in_` is missing — prefer to keep the named-features path working rather than relying on that fallback.
