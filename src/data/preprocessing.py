import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import joblib
from pathlib import Path


RARE_CLASSES = ["HYA", "IEO", "AST", "CEN"]

DROP_COLUMNS = [
    "id", "spkid", "full_name", "pdes", "name", "prefix",
    "orbit_id", "equinox", "epoch", "epoch_cal", "tp_cal",
    "q", "ad", "n", "per", "per_y", "moid_ld",
    "diameter_sigma", "albedo",
    # New JPL columns we explicitly drop because coverage is too low
    # to be useful (<0.1% of rows). The model would just learn the
    # median value, which is noise.
    "G", "BV", "UB", "IR",
    # Spectral types: high signal but only 0.1% coverage — would need
    # one-hot encoding with a large "Unknown" bucket. Skip for now to
    # focus on the n_obs_used/data_arc/condition_code gold trio.
    "spec_B", "spec_T",
    # Geometric/timing fields with near-zero signal (max_signal < 0.07
    # against all three targets). They describe where the asteroid is
    # in its orbit at the snapshot epoch — not what it *is*.
    "om", "w", "tp", "epoch_mjd",
    # H_sigma was useful when H was a target (heteroscedastic loss);
    # now that H is an input feature, H_sigma adds noise.
    "H_sigma",
    # NOTE: rot_per stays in raw — it is now the 4th regression target,
    # extracted by preprocess() before this drop.
]

TARGET_CLASS = "class"
TARGET_H = "H"
TARGET_DIAMETER = "diameter"
TARGET_ALBEDO = "albedo"
TARGET_ROT_PER = "rot_per"

JUPITER_A = 5.2044
MARS_A = 1.523679
SATURN_A = 9.5826
NEPTUNE_A = 30.0699

# (P_asteroid / P_planet, planet a, output column).
# At exact resonance: a_res = a_planet * (P_ratio)^(2/3) by Kepler's third law.
RESONANCES = [
    (1.0,   MARS_A,    "res_mars_co"),   # Mars co-orbitals (e.g. 5261 Eureka)
    (1/3,   JUPITER_A, "res_jup_3_1"),   # Kirkwood gap ~2.50 AU
    (2/5,   JUPITER_A, "res_jup_5_2"),   # Kirkwood gap ~2.82 AU
    (1/2,   JUPITER_A, "res_jup_2_1"),   # Kirkwood gap ~3.28 AU
    (2/3,   JUPITER_A, "res_jup_3_2"),   # Hildas ~3.97 AU
    (3/2,   NEPTUNE_A, "res_nep_2_3"),   # Plutinos ~39.4 AU
    (2.0,   NEPTUNE_A, "res_nep_1_2"),   # Twotinos ~47.7 AU
]


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def compute_tisserand(df: pd.DataFrame, a_planet: float = JUPITER_A) -> pd.Series:
    a = df["a"]
    e = df["e"]
    i_rad = np.radians(df["i"])
    return a_planet / a + 2 * np.cos(i_rad) * np.sqrt(a / a_planet * (1 - e**2))


def compute_resonance_distance(a: pd.Series, period_ratio: float, a_planet: float) -> pd.Series:
    a_res = a_planet * period_ratio ** (2 / 3)
    return (a - a_res) / a_res


def drop_corrupt_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    valid_a = df["a"].notna() & (df["a"] > 0)
    valid_e = df["e"].notna() & (df["e"] >= 0) & (df["e"] < 1)
    valid_i = df["i"].notna() & (df["i"] >= 0) & (df["i"] <= 180)
    valid_rms = df["rms"].isna() | (df["rms"] <= 10)
    keep = valid_a & valid_e & valid_i & valid_rms
    dropped = n_before - keep.sum()
    if dropped > 0:
        print(f"Dropped {dropped} corrupt rows (impossible orbital elements or rms > 10)")
    return df.loc[keep].reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    df = drop_corrupt_rows(df)
    # H is now a model INPUT, not a target. Drop the ~0.7% of rows where H is
    # missing — we need it to predict diameter/albedo, and there is no
    # principled way to fill it (the whole point is that H is cheap to measure).
    h_present = df[TARGET_H].notna()
    dropped_h = (~h_present).sum()
    if dropped_h > 0:
        print(f"Dropped {dropped_h} rows without measured H (now a required feature)")
    df = df.loc[h_present].reset_index(drop=True)

    df["neo"] = df["neo"].fillna("N").map({"Y": 1, "N": 0}).astype(float)
    df["pha"] = df["pha"].fillna("N").map({"Y": 1, "N": 0}).astype(float)

    # moid has 100 % coverage in the refreshed JPL catalogue — no missing
    # indicator needed.
    df["moid"] = df["moid"].fillna(df["moid"].max())

    sigma_cols = [c for c in df.columns if c.startswith("sigma_")]
    for col in sigma_cols:
        df[col] = df[col].fillna(df[col].median())

    # Observation-quality features pulled fresh from JPL (see fetch_jpl_full.py).
    # n_obs_used carries the strongest single signal for H (Spearman ρ=-0.803).
    # data_arc and condition_code are secondary but still informative.
    if "n_obs_used" in df.columns:
        df["n_obs_used"] = np.log1p(df["n_obs_used"].fillna(0))
    if "data_arc" in df.columns:
        df["data_arc"] = np.log1p(df["data_arc"].fillna(df["data_arc"].median()))
    if "condition_code" in df.columns:
        df["condition_code"] = pd.to_numeric(df["condition_code"], errors="coerce")
        df["condition_code"] = df["condition_code"].fillna(df["condition_code"].median())

    # Only tisserand_j survives the feature-importance audit: it discriminates
    # asteroids from Jupiter-family comets, with a non-trivial value across the
    # whole catalogue. The other Tisserand parameters and all resonance-distance
    # features were affine functions of `a` and offered zero new signal beyond
    # what a deep MLP can derive from `a` itself.
    df["tisserand_j"] = compute_tisserand(df, JUPITER_A)

    target_class = df[TARGET_CLASS].copy()
    target_class = target_class.replace({cls: "Rare" for cls in RARE_CLASSES})

    target_diameter = np.log1p(df[TARGET_DIAMETER].copy())
    # Albedo physically in (0, 1]; train head in log-space (range ~ [-7, 0]).
    # Clip tiny invalid values to avoid log(0) → -inf for the (very few) zero rows.
    target_albedo = np.log(df[TARGET_ALBEDO].copy().clip(lower=1e-4))
    # Rotation period in hours: ~2 % coverage, log-normal distribution
    # spanning [0.001, 4800] h. Regress in natural log space (range ~ [-7, 8]).
    target_rot_per = np.log(df[TARGET_ROT_PER].copy().clip(lower=0.01))

    # NOTE: H stays in df — it is now a model input feature, not a target.
    df = df.drop(
        columns=DROP_COLUMNS + [TARGET_CLASS, TARGET_DIAMETER, TARGET_ROT_PER],
        errors="ignore",
    )

    sigma_cols_present = [c for c in df.columns if c.startswith("sigma_")]
    for col in sigma_cols_present:
        df[col] = np.log1p(df[col].clip(lower=0))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df, target_class, target_diameter, target_albedo, target_rot_per


def build_splits(
    raw_path: str,
    output_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    features, target_class, target_diameter, target_albedo, target_rot_per = preprocess(df)

    label_encoder = LabelEncoder()
    class_encoded = label_encoder.fit_transform(target_class)

    diam_mask = (~target_diameter.isna()).astype(float).values
    albedo_mask = (~target_albedo.isna()).astype(float).values
    rot_mask = (~target_rot_per.isna()).astype(float).values
    target_diameter_filled = target_diameter.fillna(0).values
    target_albedo_filled = target_albedo.fillna(0).values
    target_rot_per_filled = target_rot_per.fillna(0).values

    numeric_features = features.select_dtypes(include=[np.number])

    X_temp, X_test, y_cls_temp, y_cls_test = train_test_split(
        numeric_features, class_encoded,
        test_size=test_size,
        stratify=class_encoded,
        random_state=random_state,
    )

    idx_temp = X_temp.index
    idx_test = X_test.index

    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_cls_train, y_cls_val = train_test_split(
        X_temp, y_cls_temp,
        test_size=relative_val,
        stratify=y_cls_temp,
        random_state=random_state,
    )
    idx_train = X_train.index
    idx_val = X_val.index

    scaler = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(X_train)),
        random_state=random_state,
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    feature_names = numeric_features.columns.tolist()

    def make_df(X_scaled, idx, split_name):
        out = pd.DataFrame(X_scaled, columns=feature_names)
        out["class_label"] = class_encoded[idx]
        out["diameter_target"] = target_diameter_filled[idx]
        out["albedo_target"] = target_albedo_filled[idx]
        out["rot_target"] = target_rot_per_filled[idx]
        out["diameter_mask"] = diam_mask[idx]
        out["albedo_mask"] = albedo_mask[idx]
        out["rot_mask"] = rot_mask[idx]
        return out

    train_df = make_df(X_train_scaled, idx_train, "train")
    val_df = make_df(X_val_scaled, idx_val, "val")
    test_df = make_df(X_test_scaled, idx_test, "test")

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    # Class-conditional albedo prior (mean log albedo per class on TRAIN only).
    # The residual albedo head adds this as a bias term so the model learns
    # |class_mean - true_albedo| instead of the full target. Drastically
    # reduces the dynamic range the head has to fit.
    class_mean_log_albedo = np.zeros(len(label_encoder.classes_), dtype=np.float32)
    for k in range(len(label_encoder.classes_)):
        mask_k = (train_df["class_label"] == k) & (train_df["albedo_mask"] == 1)
        vals = train_df.loc[mask_k, "albedo_target"].values
        if len(vals) > 0:
            class_mean_log_albedo[k] = float(np.mean(vals))
        else:
            class_mean_log_albedo[k] = float(np.log(0.1))  # fallback ≈ typical
    joblib.dump(class_mean_log_albedo, output_dir / "class_albedo_prior.joblib")
    print("Class-mean log(albedo) (residual baseline):")
    for k, name in enumerate(label_encoder.classes_):
        print(f"  {name}: {class_mean_log_albedo[k]:+.3f}  (albedo ≈ {np.exp(class_mean_log_albedo[k]):.3f})")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {label_encoder.classes_.tolist()}")
    print(f"Diameter coverage - train: {train_df['diameter_mask'].mean():.1%}, "
          f"val: {val_df['diameter_mask'].mean():.1%}, "
          f"test: {test_df['diameter_mask'].mean():.1%}")
    print(f"Albedo coverage   - train: {train_df['albedo_mask'].mean():.1%}, "
          f"val: {val_df['albedo_mask'].mean():.1%}, "
          f"test: {test_df['albedo_mask'].mean():.1%}")
    print(f"RotPer coverage   - train: {train_df['rot_mask'].mean():.1%}, "
          f"val: {val_df['rot_mask'].mean():.1%}, "
          f"test: {test_df['rot_mask'].mean():.1%}")

    return train_df, val_df, test_df, scaler, label_encoder


if __name__ == "__main__":
    build_splits("data/raw/dataset.csv", "data/processed")
