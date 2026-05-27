import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.express as px

from src.models.mtl_model import AsteroidMTLModel


@st.cache_resource
def load_model():
    label_encoder = joblib.load("../data/processed/label_encoder.joblib")
    scaler = joblib.load("../data/processed/scaler.joblib")
    n_classes = len(label_encoder.classes_)
    n_features = len(scaler.mean_)

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
    return model, scaler, label_encoder


def compute_tisserand(a, e, i_deg):
    a_j = 5.2044
    i_rad = np.radians(i_deg)
    return a_j / a + 2 * np.cos(i_rad) * np.sqrt(a / a_j * (1 - e**2))


st.set_page_config(page_title="Asteroid MTL Predictor", layout="wide")
st.title("Asteroid Classification & Physical Property Prediction")
st.markdown("Multitask neural network trained on NASA JPL asteroid data.")

model, scaler, label_encoder = load_model()

st.sidebar.header("Orbital Parameters")

e = st.sidebar.slider("Eccentricity (e)", 0.0, 1.0, 0.15, 0.01)
a = st.sidebar.slider("Semi-major axis (a) [AU]", 0.5, 30.0, 2.7, 0.1)
i = st.sidebar.slider("Inclination (i) [deg]", 0.0, 90.0, 10.0, 0.5)
om = st.sidebar.slider("Long. ascending node (om) [deg]", 0.0, 360.0, 180.0, 1.0)
w = st.sidebar.slider("Arg. perihelion (w) [deg]", 0.0, 360.0, 180.0, 1.0)
ma = st.sidebar.slider("Mean anomaly (ma) [deg]", 0.0, 360.0, 180.0, 1.0)

if st.sidebar.button("Predict", type="primary"):
    tisserand = compute_tisserand(a, e, i)

    feature_names = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None

    input_dict = {
        "e": e, "a": a, "i": i, "om": om, "w": w, "ma": ma,
        "tisserand_j": tisserand,
    }

    if feature_names is not None:
        input_values = []
        for name in feature_names:
            input_values.append(input_dict.get(name, 0.0))
        raw = np.array([input_values])
    else:
        raw = np.array([[e, a, i, om, w, ma, tisserand, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    scaled = scaler.transform(raw)
    x = torch.FloatTensor(scaled)

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

    st.subheader("Input Summary")
    st.json({
        "eccentricity": e,
        "semi_major_axis_au": a,
        "inclination_deg": i,
        "tisserand_parameter": round(tisserand, 4),
    })
