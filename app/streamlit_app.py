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
from src.data.preprocessing import JUPITER_A


USER_INPUT_FEATS = ["e", "a", "i", "ma", "H"]
DERIVED_FEATS = ["tisserand_j"]
ORBITAL_FEATS = USER_INPUT_FEATS + DERIVED_FEATS
KNN_K = 100
CUSTOM_LABEL = "Custom (k-NN inferred)"

RARE_REMAP = {"HYA": "Rare", "IEO": "Rare", "AST": "Rare", "CEN": "Rare"}

PRESET_ASTEROID_PDES = [
    "1",        # Ceres
    "4",        # Vesta
    "65",       # Cybele
    "433",      # Eros
    "434",      # Hungaria
    "588",      # Achilles
    "1862",     # Apollo
    "2060",     # Chiron
    "2062",     # Aten
    "5261",     # Eureka
    "99942",    # Apophis
    "134340",   # Pluto
]


@st.cache_resource
def load_model():
    label_encoder = joblib.load("../data/processed/label_encoder.joblib")
    scaler = joblib.load("../data/processed/scaler.joblib")
    n_classes = len(label_encoder.classes_)
    n_features = scaler.n_features_in_
    feat_names = list(scaler.feature_names_in_)

    class_albedo_prior = joblib.load("../data/processed/class_albedo_prior.joblib")
    model = AsteroidMTLModel(
        n_features=n_features,
        n_classes=n_classes,
        backbone_layers=[512, 256, 128, 64],
        backbone_dropouts=[0.3, 0.3, 0.2, 0.2],
        head_hidden=32,
        head_dropout=0.1,
        class_albedo_prior=class_albedo_prior,
        rot_mdn_components=5,
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

    rot_mask_train = train["rot_mask"].values.astype(bool)
    global_rot_log_values = train.loc[rot_mask_train, "rot_target"].values
    global_rot_log_median = float(np.median(global_rot_log_values))
    global_rot_log_q25 = float(np.percentile(global_rot_log_values, 25))
    global_rot_log_q75 = float(np.percentile(global_rot_log_values, 75))

    display_cols = ["pdes", "name", "class", "e", "a", "i", "om", "w", "ma",
                    "H", "diameter", "albedo", "rot_per"]
    raw = pd.read_csv("../data/raw/dataset.csv", low_memory=False, usecols=display_cols)
    raw["pdes"] = raw["pdes"].astype(str).str.strip()
    raw["taxonomy"] = raw["class"].replace(RARE_REMAP)
    preset_rows = raw[raw["pdes"].isin(PRESET_ASTEROID_PDES)].set_index("pdes")
    preset_rows = preset_rows.loc[[p for p in PRESET_ASTEROID_PDES if p in preset_rows.index]]


    from src.data.preprocessing import preprocess, drop_corrupt_rows
    full_raw = pd.read_csv("../data/raw/dataset.csv", low_memory=False)
    full_raw["pdes_str"] = full_raw["pdes"].astype(str).str.strip()
    full_clean = drop_corrupt_rows(full_raw)
    full_features, _, _, _, _ = preprocess(full_raw.copy())
    preset_pdes_set = set(PRESET_ASTEROID_PDES)
    preset_clean_idx = full_clean.index[full_clean["pdes_str"].isin(preset_pdes_set)].tolist()
    preset_features_aligned = full_features.loc[preset_clean_idx, feat_names]
    preset_scaled_matrix = scaler.transform(preset_features_aligned.values).astype(np.float32)
    preset_scaled = {
        full_clean.loc[idx, "pdes_str"]: row
        for idx, row in zip(preset_clean_idx, preset_scaled_matrix)
    }

    # Raw (pre-scaler) feature values for presets — exactly what scaler.transform
    # receives. Used to show the catalogued 24-feature vector and to seed the
    # manual "All features" tab.
    preset_raw_matrix = preset_features_aligned.values.astype(np.float64)
    preset_raw = {
        full_clean.loc[idx, "pdes_str"]: row
        for idx, row in zip(preset_clean_idx, preset_raw_matrix)
    }

    # Per-feature training statistics in the pre-scaler space (defaults / hints
    # for manual entry).
    feat_median = full_features[feat_names].median().values.astype(np.float64)
    feat_min = full_features[feat_names].min().values.astype(np.float64)
    feat_max = full_features[feat_names].max().values.astype(np.float64)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feat_names": feat_names,
        "X_scaled": X_scaled,
        "preset_scaled": preset_scaled,
        "preset_raw": preset_raw,
        "feat_median": feat_median,
        "feat_min": feat_min,
        "feat_max": feat_max,
        "train_class_labels": train_class_labels,
        "nn_index": nn_index,
        "orbital_idx": orbital_idx,
        "other_idx": other_idx,
        "preset_rows": preset_rows,
        "global_rot_log_median": global_rot_log_median,
        "global_rot_log_q25": global_rot_log_q25,
        "global_rot_log_q75": global_rot_log_q75,
    }


def tisserand_scalar(a, e, i_deg, a_planet):
    i_rad = np.radians(i_deg)
    return a_planet / a + 2 * np.cos(i_rad) * np.sqrt(a / a_planet * (1 - e**2))


def compute_derived(a, e, i_deg) -> dict:
    """Reproduce src/data/preprocessing.py feature engineering for one (a,e,i)."""
    return {"tisserand_j": tisserand_scalar(a, e, i_deg, JUPITER_A)}


def predict_with_knn_fill(user_raw: dict, ctx) -> tuple[np.ndarray, np.ndarray]:
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

    distances, ind = ctx["nn_index"].kneighbors(
        orbital_scaled.reshape(1, -1), n_neighbors=KNN_K, return_distance=True,
    )
    distances = distances[0]
    ind = ind[0]
    weights = 1.0 / (distances + 1e-6)
    weights /= weights.sum()
    other_filled = (ctx["X_scaled"][ind][:, other_idx] * weights[:, None]).sum(axis=0)

    scaled_row = np.empty(len(feat_names), dtype=np.float32)
    scaled_row[orbital_idx] = orbital_scaled
    scaled_row[other_idx] = other_filled
    return scaled_row, ind


st.set_page_config(page_title="Asteroid MTL Predictor", layout="wide")
st.title("Asteroid Classification & Physical Property Prediction")
st.markdown("Multitask neural network trained on NASA JPL asteroid data.")

ctx = load_model()
model = ctx["model"]
label_encoder = ctx["label_encoder"]
preset_rows = ctx["preset_rows"]

PRESETS: dict[str, str | None] = {CUSTOM_LABEL: None}
for pdes, row in preset_rows.iterrows():
    name = row["name"] if pd.notna(row["name"]) else f"#{pdes}"
    PRESETS[f"{pdes} {name} ({row['class']})"] = pdes

DEFAULT_PRESET = next(k for k in PRESETS if PRESETS[k] == "1")
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


INPUT_MODE_STD = "Preset / Custom (k-NN)"
INPUT_MODE_ALL = "All features (manual)"

feat_names = ctx["feat_names"]
H_idx = feat_names.index("H")

# Per-feature input types for the manual "All features" mode.
BOOL_FEATS = {"neo", "pha"}          # catalogue flags, 0/1
INT_FEATS = {"condition_code"}       # JPL orbit condition code, integer 0–9

# Seed the manual feature inputs once from per-feature training medians, with
# the correct Python type so the matching widget accepts the session value.
for j, name in enumerate(feat_names):
    med = float(ctx["feat_median"][j])
    if name in BOOL_FEATS:
        st.session_state.setdefault(f"feat_{name}", bool(round(med)))
    elif name in INT_FEATS:
        st.session_state.setdefault(f"feat_{name}", int(round(med)))
    else:
        st.session_state.setdefault(f"feat_{name}", med)


st.sidebar.header("Input mode")
input_mode = st.sidebar.radio(
    "How features are provided",
    [INPUT_MODE_STD, INPUT_MODE_ALL],
    key="input_mode",
    help=(
        "Preset / Custom (k-NN): set only the orbital elements; for a catalogue "
        "preset the real feature vector is used, otherwise the non-orbital "
        "features are inferred via k-NN.\n\n"
        "All features (manual): type every one of the 24 model inputs by hand."
    ),
)

# These are populated by whichever mode is active and consumed by Predict.
e = a = i = ma = H = None
derived = None

if input_mode == INPUT_MODE_STD:
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

    e  = st.sidebar.number_input("Eccentricity (e)",      min_value=0.0, max_value=0.99,  key="e",  step=0.01, format="%.4f", on_change=mark_custom)
    a  = st.sidebar.number_input("Semi-major axis a [AU]", min_value=0.1, max_value=200.0, key="a",  step=0.1,  format="%.4f", on_change=mark_custom)
    i  = st.sidebar.number_input("Inclination i [deg]",    min_value=0.0, max_value=180.0, key="i",  step=0.5,  format="%.2f", on_change=mark_custom)
    ma = st.sidebar.number_input("Mean anomaly ma [deg]",  min_value=0.0, max_value=360.0, key="ma", step=1.0,  format="%.2f", on_change=mark_custom)
    st.sidebar.markdown("---")
    H = st.sidebar.number_input(
        "Absolute magnitude H [mag]",
        min_value=-5.0, max_value=35.0, key="H", step=0.1, format="%.2f",
        on_change=mark_custom,
        help="Observed absolute magnitude — cheap to measure (any survey gives V; "
             "H follows from V and orbit). The model uses H as a known input to "
             "predict diameter and albedo.",
    )
else:
    st.sidebar.header("All 24 features")
    st.sidebar.caption("Tick **n/a** for any feature you don't have — the training median is used instead.")
    for j, name in enumerate(feat_names):
        col_val, col_unk = st.sidebar.columns([3, 1], vertical_alignment="bottom")
        unknown = col_unk.checkbox("n/a", key=f"unk_{name}", help="Unknown — use training median")
        if name in BOOL_FEATS:
            col_val.checkbox(name, key=f"feat_{name}", disabled=unknown)
        elif name in INT_FEATS:
            col_val.number_input(
                name, key=f"feat_{name}", min_value=0, max_value=9, step=1,
                disabled=unknown,
                help=f"median ≈ {ctx['feat_median'][j]:.4g}" if unknown else None,
            )
        else:
            col_val.number_input(
                name, key=f"feat_{name}", step=0.01, format="%.4f", disabled=unknown,
                help=f"median ≈ {ctx['feat_median'][j]:.4g}" if unknown else None,
            )

if st.sidebar.button("Predict", type="primary"):
    if input_mode == INPUT_MODE_ALL:
        # Build the raw 24-vector from the manual fields; any feature ticked
        # "unknown" falls back to the training median. Then scale it.
        raw_vec = np.array([
            ctx["feat_median"][j] if st.session_state.get(f"unk_{n}") else st.session_state[f"feat_{n}"]
            for j, n in enumerate(feat_names)
        ], dtype=np.float64)
        scaled_row = ctx["scaler"].transform(raw_vec.reshape(1, -1))[0].astype(np.float32)
        neighbor_indices = None
        source = "manual"
        n_unknown = sum(bool(st.session_state.get(f"unk_{n}")) for n in feat_names)
        fill_mode = (
            f"manual entry ({24 - n_unknown}/24 set, {n_unknown} filled with train median)"
            if n_unknown else "manual entry of all 24 features"
        )
        H = float(raw_vec[H_idx])
        a = float(raw_vec[feat_names.index("a")])
        e = float(raw_vec[feat_names.index("e")])
        i = float(raw_vec[feat_names.index("i")])
        derived = {"tisserand_j": float(raw_vec[feat_names.index("tisserand_j")])}
    else:
        derived = compute_derived(a, e, i)
        user_raw = {"e": e, "a": a, "i": i, "ma": ma, "H": H, **derived}

        preset_pdes = PRESETS.get(st.session_state.preset)
        if preset_pdes is not None and preset_pdes in ctx["preset_scaled"]:
            scaled_row = ctx["preset_scaled"][preset_pdes]
            neighbor_indices = None
            source = "preset"
            fill_mode = f"real 24 features (catalogued asteroid {preset_pdes})"
        else:
            scaled_row, neighbor_indices = predict_with_knn_fill(user_raw, ctx)
            source = "knn"
            fill_mode = f"k-NN over {KNN_K} nearest training asteroids"

    x = torch.FloatTensor(scaled_row.reshape(1, -1))
    with torch.no_grad():
        outputs = model(x)

    probs = torch.softmax(outputs["class_logits"], dim=1).squeeze().numpy()
    pred_class = label_encoder.classes_[probs.argmax()]
    pred_diam_learned = np.expm1(outputs["diameter_pred"].item())
    pred_albedo = float(np.clip(np.exp(outputs["albedo_pred"].item()), 1e-3, 1.0))

    # Trust the MDN rotation head only when the model sees a real, in-distribution
    # feature vector (preset or full manual entry). For k-NN-filled inputs fall
    # back to the global train median + IQR.
    use_global_rot = source == "knn"
    if use_global_rot:
        pred_rot_per = float(np.exp(ctx["global_rot_log_median"]))
        rot_low = float(np.exp(ctx["global_rot_log_q25"]))
        rot_high = float(np.exp(ctx["global_rot_log_q75"]))
        rot_display = f"{pred_rot_per:.2f} h"
        rot_delta = f"global IQR: {rot_low:.1f}–{rot_high:.1f} h"
    else:
        pred_rot_per = float(np.exp(outputs["rot_pred"].item()))
        log_pi_np = outputs["rot_log_pi"].squeeze(0).numpy()
        mu_np = outputs["rot_mu"].squeeze(0).numpy()
        sorted_modes = np.argsort(log_pi_np)[::-1]
        primary_mode = sorted_modes[0]
        secondary_mode = sorted_modes[1] if len(sorted_modes) > 1 else primary_mode
        rot_primary_prob = float(np.exp(log_pi_np[primary_mode]))
        rot_secondary_value = float(np.exp(mu_np[secondary_mode]))
        rot_secondary_prob = float(np.exp(log_pi_np[secondary_mode]))
        rot_display = f"{pred_rot_per:.2f} h ({rot_primary_prob*100:.0f}%)"
        rot_delta = f"alt: {rot_secondary_value:.1f} h ({rot_secondary_prob*100:.0f}%)"

    pred_diam_physical = 1329.0 / np.sqrt(pred_albedo) * 10 ** (-H / 5)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Predicted Class", pred_class, f"{probs.max()*100:.1f}% confidence")
    c2.metric("Geometric albedo", f"{pred_albedo:.3f}")
    c3.metric("Diameter (physical)", f"{pred_diam_physical:.2f} km",
              delta=f"learned head: {pred_diam_learned:.2f} km", delta_color="off")
    c4.metric("Rotation period", rot_display, delta=rot_delta, delta_color="off")
    c5.metric("Input H (known)", f"{H:.2f}")

    preset_pdes = PRESETS.get(st.session_state.preset) if source == "preset" else None
    if preset_pdes is not None:
        true_row = preset_rows.loc[preset_pdes]
        def _fmt(v, suffix="", places=2):
            return f"{v:.{places}f}{suffix}" if pd.notna(v) else "—"
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric(f"True class — {true_row['name']}", true_row["class"])
        t2.metric("True albedo", _fmt(true_row["albedo"], places=3))
        t3.metric("True diameter", _fmt(true_row["diameter"], " km"))
        t4.metric("True rot_per", _fmt(true_row.get("rot_per"), " h") if "rot_per" in true_row.index else "—")
        t5.metric("True H", _fmt(true_row["H"]))

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
            "eccentricity": round(e, 4),
            "semi_major_axis_au": round(a, 4),
            "inclination_deg": round(i, 2),
            "tisserand_jupiter": round(derived["tisserand_j"], 3),
            "feature_fill_strategy": fill_mode,
            "diameter_formula": "D[km] = 1329 / sqrt(albedo) · 10^(-H/5)",
        })
    with col_b:
        if source == "knn":
            st.subheader(f"k-NN neighbors (k={KNN_K})")
            st.caption("Class distribution of training asteroids used to infer the non-orbital features.")
            neighbor_cls = [label_encoder.classes_[c] for c in ctx["train_class_labels"][neighbor_indices]]
            st.bar_chart(pd.Series(neighbor_cls).value_counts())
        elif source == "preset":
            st.subheader("Real catalogue features (all 24)")
            st.caption(
                "Prediction uses the asteroid's actual measured values — "
                "n_obs_used, data_arc, condition_code, sigma_* and the rest — "
                "no k-NN approximation."
            )
            feat_df = pd.DataFrame({
                "feature": feat_names,
                "value": np.round(ctx["preset_raw"][preset_pdes], 4),
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)
        else:  # manual
            st.subheader("Manual feature vector (all 24)")
            st.caption(
                "Values fed to the model. source = 'user' (you typed it) or "
                "'median' (ticked unknown, filled with the training median)."
            )
            feat_df = pd.DataFrame({
                "feature": feat_names,
                "value": np.round(raw_vec, 4),
                "source": [
                    "median" if st.session_state.get(f"unk_{n}") else "user"
                    for n in feat_names
                ],
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)
