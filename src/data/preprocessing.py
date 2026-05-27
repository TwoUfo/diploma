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
]

TARGET_CLASS = "class"
TARGET_H = "H"
TARGET_DIAMETER = "diameter"

JUPITER_A = 5.2044


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def compute_tisserand(df: pd.DataFrame) -> pd.Series:
    a = df["a"]
    e = df["e"]
    i_rad = np.radians(df["i"])
    return JUPITER_A / a + 2 * np.cos(i_rad) * np.sqrt(a / JUPITER_A * (1 - e**2))


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


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = drop_corrupt_rows(df)

    df["neo"] = df["neo"].fillna("N").map({"Y": 1, "N": 0}).astype(float)
    df["pha"] = df["pha"].fillna("N").map({"Y": 1, "N": 0}).astype(float)

    df["moid_missing"] = df["moid"].isna().astype(float)
    df["moid"] = df["moid"].fillna(df["moid"].max())

    sigma_cols = [c for c in df.columns if c.startswith("sigma_")]
    for col in sigma_cols:
        df[col] = df[col].fillna(df[col].median())

    df["tisserand_j"] = compute_tisserand(df)

    target_class = df[TARGET_CLASS].copy()
    target_class = target_class.replace({cls: "Rare" for cls in RARE_CLASSES})

    target_h = df[TARGET_H].copy()
    target_diameter = df[TARGET_DIAMETER].copy()
    target_diameter = np.log1p(target_diameter)

    df = df.drop(columns=DROP_COLUMNS + [TARGET_CLASS, TARGET_H, TARGET_DIAMETER], errors="ignore")

    sigma_cols_present = [c for c in df.columns if c.startswith("sigma_")]
    for col in sigma_cols_present:
        df[col] = np.log1p(df[col].clip(lower=0))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df, target_class, target_h, target_diameter


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
    features, target_class, target_h, target_diameter = preprocess(df)

    label_encoder = LabelEncoder()
    class_encoded = label_encoder.fit_transform(target_class)

    h_mask = (~target_h.isna()).astype(float).values
    diam_mask = (~target_diameter.isna()).astype(float).values
    target_h_filled = target_h.fillna(0).values
    target_diameter_filled = target_diameter.fillna(0).values

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
        out["h_target"] = target_h_filled[idx]
        out["diameter_target"] = target_diameter_filled[idx]
        out["h_mask"] = h_mask[idx]
        out["diameter_mask"] = diam_mask[idx]
        return out

    train_df = make_df(X_train_scaled, idx_train, "train")
    val_df = make_df(X_val_scaled, idx_val, "val")
    test_df = make_df(X_test_scaled, idx_test, "test")

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {label_encoder.classes_.tolist()}")
    print(f"Diameter coverage - train: {train_df['diameter_mask'].mean():.1%}, "
          f"val: {val_df['diameter_mask'].mean():.1%}, "
          f"test: {test_df['diameter_mask'].mean():.1%}")

    return train_df, val_df, test_df, scaler, label_encoder


if __name__ == "__main__":
    build_splits("data/raw/dataset.csv", "data/processed")
