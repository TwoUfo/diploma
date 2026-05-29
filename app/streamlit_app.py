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
from src.data.preprocessing import (
    JUPITER_A, MARS_A, SATURN_A, NEPTUNE_A, RESONANCES,
)


USER_INPUT_FEATS = ["e", "a", "i", "om", "w", "ma"]
DERIVED_FEATS = (
    ["tisserand_j", "tisserand_mars", "tisserand_saturn", "tisserand_neptune"]
    + [name for _, _, name in RESONANCES]
)
ORBITAL_FEATS = USER_INPUT_FEATS + DERIVED_FEATS
KNN_K = 100
CUSTOM_LABEL = "Custom (k-NN inferred)"

RARE_REMAP = {"HYA": "Rare", "IEO": "Rare", "AST": "Rare", "CEN": "Rare"}

# Curated set of real, well-known asteroids loaded from the raw catalogue at startup.
# Each one is recognisable, scientifically interesting, and spans the taxonomic range.
PRESET_ASTEROID_PDES = [
    "1",        # Ceres — largest MBA / dwarf planet
    "4",        # Vesta — second-largest MBA, basaltic surface
    "65",       # Cybele — namesake of OMB Cybele group
    "433",      # Eros — first NEO discovered, NEAR landing site
    "434",      # Hungaria — namesake of IMB Hungaria family
    "588",      # Achilles — first Jupiter Trojan
    "1862",     # Apollo — namesake of APO class
    "2060",     # Chiron — first known Centaur
    "2062",     # Aten — namesake of ATE class
    "5261",     # Eureka — first Mars Trojan
    "99942",    # Apophis — past Earth-impact-risk poster child
    "134340",   # Pluto — dwarf planet, TNO
]


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

    raw_cols = ["pdes", "name", "class", "e", "a", "i", "om", "w", "ma",
                "H", "diameter", "albedo"]
    raw = pd.read_csv("../data/raw/dataset.csv", low_memory=False, usecols=raw_cols)
    raw["pdes"] = raw["pdes"].astype(str).str.strip()
    raw["taxonomy"] = raw["class"].replace(RARE_REMAP)
    preset_rows = raw[raw["pdes"].isin(PRESET_ASTEROID_PDES)].set_index("pdes")
    preset_rows = preset_rows.loc[[p for p in PRESET_ASTEROID_PDES if p in preset_rows.index]]

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
        "preset_rows": preset_rows,
    }


def tisserand_scalar(a, e, i_deg, a_planet):
    i_rad = np.radians(i_deg)
    return a_planet / a + 2 * np.cos(i_rad) * np.sqrt(a / a_planet * (1 - e**2))


def compute_derived(a, e, i_deg) -> dict:
    """Reproduce src/data/preprocessing.py feature engineering for a single (a,e,i)."""
    out = {
        "tisserand_j":       tisserand_scalar(a, e, i_deg, JUPITER_A),
        "tisserand_mars":    tisserand_scalar(a, e, i_deg, MARS_A),
        "tisserand_saturn":  tisserand_scalar(a, e, i_deg, SATURN_A),
        "tisserand_neptune": tisserand_scalar(a, e, i_deg, NEPTUNE_A),
    }
    for ratio, a_planet, name in RESONANCES:
        a_res = a_planet * ratio ** (2 / 3)
        out[name] = (a - a_res) / a_res
    return out


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
preset_rows = ctx["preset_rows"]

# PRESETS maps display label → pdes (None for Custom). Real asteroids only —
# no abstract class-median entries.
PRESETS: dict[str, str | None] = {CUSTOM_LABEL: None}
for pdes, row in preset_rows.iterrows():
    name = row["name"] if pd.notna(row["name"]) else f"#{pdes}"
    PRESETS[f"{pdes} {name} ({row['class']})"] = pdes

DEFAULT_PRESET = next(k for k in PRESETS if PRESETS[k] == "1")  # Ceres
for k in USER_INPUT_FEATS:
    st.session_state.setdefault(k, float(preset_rows.loc["1", k]))
st.session_state.setdefault("preset", DEFAULT_PRESET)


def apply_preset():
    pdes = PRESETS.get(st.session_state.preset)
    if pdes is None:
        return
    row = preset_rows.loc[pdes]
    for k in USER_INPUT_FEATS:
        st.session_state[k] = float(row[k])


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
        f"Pick a real named asteroid from the catalogue to load its orbital "
        f"elements. Editing any value switches to '{CUSTOM_LABEL}'. "
        f"In every case non-orbital features are inferred via {KNN_K}-NN."
    ),
)

e  = st.sidebar.number_input("Eccentricity (e)",              min_value=0.0, max_value=0.99,  key="e",  step=0.01, format="%.4f", on_change=mark_custom)
a  = st.sidebar.number_input("Semi-major axis a [AU]",        min_value=0.1, max_value=200.0, key="a",  step=0.1,  format="%.4f", on_change=mark_custom)
i  = st.sidebar.number_input("Inclination i [deg]",           min_value=0.0, max_value=180.0, key="i",  step=0.5,  format="%.2f", on_change=mark_custom)
om = st.sidebar.number_input("Long. ascending node om [deg]", min_value=0.0, max_value=360.0, key="om", step=1.0,  format="%.2f", on_change=mark_custom)
w  = st.sidebar.number_input("Arg. perihelion w [deg]",       min_value=0.0, max_value=360.0, key="w",  step=1.0,  format="%.2f", on_change=mark_custom)
ma = st.sidebar.number_input("Mean anomaly ma [deg]",         min_value=0.0, max_value=360.0, key="ma", step=1.0,  format="%.2f", on_change=mark_custom)

if st.sidebar.button("Predict", type="primary"):
    derived = compute_derived(a, e, i)
    user_raw = {"e": e, "a": a, "i": i, "om": om, "w": w, "ma": ma, **derived}

    scaled_row, neighbor_indices = predict_with_knn_fill(user_raw, ctx)
    fill_mode = f"k-NN over {KNN_K} nearest training asteroids"

    x = torch.FloatTensor(scaled_row.reshape(1, -1))
    with torch.no_grad():
        outputs = model(x)

    probs = torch.softmax(outputs["class_logits"], dim=1).squeeze().numpy()
    pred_class = label_encoder.classes_[probs.argmax()]
    pred_h = outputs["h_pred"].item()
    pred_diam_learned = np.expm1(outputs["diameter_pred"].item())
    pred_albedo = float(np.clip(np.exp(outputs["albedo_pred"].item()), 1e-3, 1.0))
    # Closed-form diameter from H and albedo via standard asteroid magnitude formula:
    #   D[km] = 1329 / sqrt(p_v) · 10^(-H/5)
    pred_diam_physical = 1329.0 / np.sqrt(pred_albedo) * 10 ** (-pred_h / 5)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Class", pred_class, f"{probs.max()*100:.1f}% confidence")
    c2.metric("Abs. Magnitude (H)", f"{pred_h:.2f}")
    c3.metric("Geometric albedo", f"{pred_albedo:.3f}")
    c4.metric("Diameter (physical)", f"{pred_diam_physical:.2f} km",
              delta=f"learned head: {pred_diam_learned:.2f} km", delta_color="off")

    preset_pdes = PRESETS.get(st.session_state.preset)
    if preset_pdes is not None:
        true_row = preset_rows.loc[preset_pdes]
        def _fmt(v, suffix="", places=2):
            return f"{v:.{places}f}{suffix}" if pd.notna(v) else "—"
        t1, t2, t3, t4 = st.columns(4)
        t1.metric(f"True class — {true_row['name']}", true_row["class"])
        t2.metric("True H", _fmt(true_row["H"]))
        t3.metric("True albedo", _fmt(true_row["albedo"], places=3))
        t4.metric("True diameter", _fmt(true_row["diameter"], " km"))

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
            "tisserand_jupiter": round(derived["tisserand_j"], 3),
            "tisserand_mars":    round(derived["tisserand_mars"], 3),
            "tisserand_neptune": round(derived["tisserand_neptune"], 3),
            "res_mars_co_distance":  round(derived["res_mars_co"], 4),
            "res_jup_3_2_distance":  round(derived["res_jup_3_2"], 4),
            "res_nep_2_3_distance":  round(derived["res_nep_2_3"], 4),
            "feature_fill_strategy": fill_mode,
            "diameter_formula": "D[km] = 1329 / sqrt(albedo) · 10^(-H/5)",
        })
    with col_b:
        st.subheader(f"k-NN neighbors (k={KNN_K})")
        st.caption("Class distribution of training asteroids used to infer the 18 non-orbital features.")
        neighbor_cls = [label_encoder.classes_[c] for c in ctx["train_class_labels"][neighbor_indices]]
        st.bar_chart(pd.Series(neighbor_cls).value_counts())
