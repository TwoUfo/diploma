import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

from src.models.mtl_model import AsteroidMTLModel


USER_INPUT_FEATS = ["e", "a", "i", "om", "w", "ma"]
ORBITAL_FEATS = USER_INPUT_FEATS + ["tisserand_j"]
KNN_K = 100
CUSTOM_LABEL = "Custom (k-NN inferred)"

CLASS_DESCRIPTIONS = {
    "MBA":  "Main Belt Asteroid",
    "IMB":  "Inner Main Belt",
    "OMB":  "Outer Main Belt",
    "MCA":  "Mars-Crosser",
    "AMO":  "Amor NEO",
    "APO":  "Apollo NEO",
    "ATE":  "Aten NEO",
    "TJN":  "Jupiter Trojan",
    "TNO":  "Trans-Neptunian Object",
    "Rare": "Rare (HYA/IEO/AST/CEN merged)",
}


@st.cache_resource
def load_model():
    label_encoder = joblib.load("../data/processed/label_encoder.joblib")
    scaler = joblib.load("../data/processed/scaler.joblib")
    n_classes = len(label_encoder.classes_)
    n_features = scaler.n_features_in_
    feat_names = list(scaler.feature_names_in_)

    model = AsteroidMTLModel(
        n_features=n_features,
        n_classes=n_classes,
        backbone_layers=[512, 256, 128, 64],
        backbone_dropouts=[0.3, 0.3, 0.2, 0.2],
        head_hidden=32,
        head_dropout=0.1,
    )
    checkpoint = torch.load("../models/best_mtl_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train = pd.read_parquet("../data/processed/train.parquet")
    X_scaled = train[feat_names].values.astype(np.float32)
    train_class_labels = train["class_label"].values

    orbital_idx = np.array([feat_names.index(n) for n in ORBITAL_FEATS])
    other_idx = np.array([j for j in range(n_features) if j not in orbital_idx])

    nn_index = NearestNeighbors(n_neighbors=KNN_K, algorithm="auto").fit(X_scaled[:, orbital_idx])

    X_raw = scaler.inverse_transform(X_scaled)
    df_raw = pd.DataFrame(X_raw, columns=feat_names)
    df_raw["cls"] = [label_encoder.classes_[c] for c in train_class_labels]
    class_medians_full = df_raw.groupby("cls")[feat_names].median()

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feat_names": feat_names,
        "X_scaled": X_scaled,
        "train_class_labels": train_class_labels,
        "nn_index": nn_index,
        "orbital_idx": orbital_idx,
        "other_idx": other_idx,
        "class_medians_full": class_medians_full,
    }


def compute_tisserand(a, e, i_deg):
    a_j = 5.2044
    i_rad = np.radians(i_deg)
    return a_j / a + 2 * np.cos(i_rad) * np.sqrt(a / a_j * (1 - e**2))


def predict_with_class_fill(user_raw: dict, cls: str, ctx) -> tuple[np.ndarray, np.ndarray | None]:
    """Preset path: take that class's median row, overlay user's orbital values, scale."""
    scaler = ctx["scaler"]
    feat_names = ctx["feat_names"]
    raw_row = ctx["class_medians_full"].loc[cls].values.copy()
    for j, name in enumerate(feat_names):
        if name in user_raw:
            raw_row[j] = user_raw[name]
    scaled = scaler.transform(raw_row.reshape(1, -1))[0].astype(np.float32)
    return scaled, None


def predict_with_knn_fill(user_raw: dict, ctx) -> tuple[np.ndarray, np.ndarray]:
    """Custom path: scale user's orbital values, fill the rest from K nearest neighbors."""
    scaler = ctx["scaler"]
    feat_names = ctx["feat_names"]
    orbital_idx = ctx["orbital_idx"]
    other_idx = ctx["other_idx"]

    placeholder = np.zeros(len(feat_names), dtype=np.float64)
    for j, name in enumerate(feat_names):
        if name in user_raw:
            placeholder[j] = user_raw[name]
    full_scaled = scaler.transform(placeholder.reshape(1, -1))[0]
    orbital_scaled = full_scaled[orbital_idx]

    _, ind = ctx["nn_index"].kneighbors(orbital_scaled.reshape(1, -1), n_neighbors=KNN_K)
    other_filled = np.median(ctx["X_scaled"][ind[0]][:, other_idx], axis=0)

    scaled_row = np.empty(len(feat_names), dtype=np.float32)
    scaled_row[orbital_idx] = orbital_scaled
    scaled_row[other_idx] = other_filled
    return scaled_row, ind[0]


st.set_page_config(page_title="Asteroid MTL Predictor", layout="wide")
st.title("Asteroid Classification & Physical Property Prediction")
st.markdown("Multitask neural network trained on NASA JPL asteroid data.")

ctx = load_model()
model = ctx["model"]
label_encoder = ctx["label_encoder"]
class_medians_full = ctx["class_medians_full"]

PRESETS: dict[str, str | None] = {CUSTOM_LABEL: None}
for cls in label_encoder.classes_:
    PRESETS[f"{cls} — {CLASS_DESCRIPTIONS.get(cls, cls)}"] = cls

DEFAULT_PRESET = next(k for k in PRESETS if (PRESETS[k] == "MBA"))
for k in USER_INPUT_FEATS:
    st.session_state.setdefault(k, float(class_medians_full.loc["MBA", k]))
st.session_state.setdefault("preset", DEFAULT_PRESET)


def apply_preset():
    cls = PRESETS.get(st.session_state.preset)
    if cls is None:
        return
    for k in USER_INPUT_FEATS:
        st.session_state[k] = float(class_medians_full.loc[cls, k])


def mark_custom():
    if st.session_state.preset != CUSTOM_LABEL:
        st.session_state.preset = CUSTOM_LABEL


st.sidebar.header("Orbital Parameters")
st.sidebar.selectbox(
    "Preset",
    list(PRESETS.keys()),
    key="preset",
    on_change=apply_preset,
    help=(
        "Pick a taxonomic class to load its median orbital elements (computed "
        "from training data at startup). Editing any value below switches to "
        f"'{CUSTOM_LABEL}', which infers non-orbital features via {KNN_K}-NN."
    ),
)

e  = st.sidebar.number_input("Eccentricity (e)",              min_value=0.0, max_value=0.99,  key="e",  step=0.01, format="%.4f", on_change=mark_custom)
a  = st.sidebar.number_input("Semi-major axis a [AU]",        min_value=0.1, max_value=200.0, key="a",  step=0.1,  format="%.4f", on_change=mark_custom)
i  = st.sidebar.number_input("Inclination i [deg]",           min_value=0.0, max_value=180.0, key="i",  step=0.5,  format="%.2f", on_change=mark_custom)
om = st.sidebar.number_input("Long. ascending node om [deg]", min_value=0.0, max_value=360.0, key="om", step=1.0,  format="%.2f", on_change=mark_custom)
w  = st.sidebar.number_input("Arg. perihelion w [deg]",       min_value=0.0, max_value=360.0, key="w",  step=1.0,  format="%.2f", on_change=mark_custom)
ma = st.sidebar.number_input("Mean anomaly ma [deg]",         min_value=0.0, max_value=360.0, key="ma", step=1.0,  format="%.2f", on_change=mark_custom)

if st.sidebar.button("Predict", type="primary"):
    tisserand = compute_tisserand(a, e, i)
    user_raw = {"e": e, "a": a, "i": i, "om": om, "w": w, "ma": ma, "tisserand_j": tisserand}

    preset_cls = PRESETS.get(st.session_state.preset)
    if preset_cls is None:
        scaled_row, neighbor_indices = predict_with_knn_fill(user_raw, ctx)
        fill_mode = f"k-NN over {KNN_K} nearest training asteroids"
    else:
        scaled_row, neighbor_indices = predict_with_class_fill(user_raw, preset_cls, ctx)
        fill_mode = f"class-{preset_cls} medians (all 25 features)"

    x = torch.FloatTensor(scaled_row.reshape(1, -1))
    with torch.no_grad():
        outputs = model(x)

    probs = torch.softmax(outputs["class_logits"], dim=1).squeeze().numpy()
    pred_class = label_encoder.classes_[probs.argmax()]
    pred_h = outputs["h_pred"].item()
    pred_diam = np.expm1(outputs["diameter_pred"].item())

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Class", pred_class, f"{probs.max()*100:.1f}% confidence")
    col2.metric("Abs. Magnitude (H)", f"{pred_h:.2f}")
    col3.metric("Diameter", f"{pred_diam:.2f} km")

    prob_df = pd.DataFrame({
        "Class": label_encoder.classes_,
        "Probability": probs,
    }).sort_values("Probability", ascending=True)

    fig = px.bar(prob_df, x="Probability", y="Class", orientation="h",
                 title="Class Probabilities")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Input Summary")
        st.json({
            "eccentricity": e,
            "semi_major_axis_au": a,
            "inclination_deg": i,
            "tisserand_parameter": round(tisserand, 4),
            "feature_fill_strategy": fill_mode,
        })
    with col_b:
        if neighbor_indices is not None:
            st.subheader(f"k-NN neighbors (k={KNN_K})")
            st.caption("Class distribution of training asteroids used to infer the 18 non-orbital features.")
            neighbor_cls = [label_encoder.classes_[c] for c in ctx["train_class_labels"][neighbor_indices]]
            st.bar_chart(pd.Series(neighbor_cls).value_counts())
        else:
            st.subheader("Preset fill")
            st.caption(f"Non-orbital features taken from the median of training asteroids in class **{preset_cls}**.")
