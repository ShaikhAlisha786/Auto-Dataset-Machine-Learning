import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Autos Price Predictor", page_icon="🚗")


@st.cache_resource
def load_artifacts():
    with open("autos_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("autos_features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("autos_dataset.csv").replace({"?": np.nan})
        df["num-of-cylinders"] = df["num-of-cylinders"].replace(
            {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}
        )
        for c in ["horsepower", "peak-rpm", "price"]:
            df[c] = pd.to_numeric(df[c])
        return df
    except FileNotFoundError:
        return None


model, features = load_artifacts()
df = load_data()

# Fallback ranges (used if autos_dataset.csv is not next to app)
FALLBACK = {
    "symboling": (-2, 3, 1), "wheel-base": (86.0, 121.0, 97.0),
    "length": (141.0, 209.0, 173.0), "width": (60.0, 72.5, 65.5),
    "height": (47.0, 60.0, 54.0), "curb-weight": (1488, 4066, 2400),
    "engine-size": (61, 326, 120), "compression-ratio": (7.0, 23.0, 9.0),
    "horsepower": (48, 288, 95), "peak-rpm": (4150, 6600, 5200),
    "city-mpg": (13, 49, 25), "highway-mpg": (16, 54, 30),
}

st.title("🚗 Autos Price Predictor")
st.write("Car ki specifications do, Decision Tree model price predict karega.")

values = {}
cols = st.columns(2)
for i, feat in enumerate(features):
    with cols[i % 2]:
        if feat == "num-of-cylinders":
            values[feat] = st.selectbox(feat, [2, 3, 4, 5, 6, 8, 12], index=2)
            continue
        if df is not None and feat in df:
            lo, hi, mid = df[feat].min(), df[feat].max(), df[feat].median()
        else:
            lo, hi, mid = FALLBACK[feat]
        if float(lo).is_integer() and float(hi).is_integer() and float(mid).is_integer():
            values[feat] = st.number_input(feat, int(lo), int(hi), int(mid), step=1)
        else:
            values[feat] = st.number_input(
                feat, float(lo), float(hi), float(round(mid, 1)), step=0.1
            )

input_df = pd.DataFrame([values], columns=features)

if st.button("Predict Price", type="primary"):
    price = float(model.predict(input_df)[0])
    st.success(f"Predicted price: **${price:,.0f}**")
    st.caption("Model test RMSE ≈ $2,846, isliye actual price is se thoda upar-neeche ho sakta hai.")

with st.expander("Input dekho"):
    st.dataframe(input_df, hide_index=True)
