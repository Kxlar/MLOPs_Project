import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Anomaly Detection Frontend", layout="centered")
st.title("Anomaly Detection - Frontend")
st.write("Upload an image and send it to the FastAPI backend (`POST /predict`).")

BACKEND = os.environ.get("BACKEND", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BACKEND}/predict"

st.caption(f"Backend: {API_URL}")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Run prediction"):
        with st.spinner("Calling backend..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
            r = requests.post(API_URL, files=files, timeout=120)

        if r.ok:
            data = r.json()
            st.success("Success ✅")

            col1, col2, col3 = st.columns(3)
            col1.metric("Anomaly score", f"{data.get('anomaly_score', 0):.4f}")
            col2.metric("Threshold", f"{data.get('threshold', 0):.4f}")
            col3.metric("Is anomaly", str(data.get("is_anomaly")))

            if data.get("heatmap_base64"):
                st.subheader("Heatmap")
                heatmap_bytes = base64.b64decode(data["heatmap_base64"])
                heatmap_img = Image.open(BytesIO(heatmap_bytes))
                st.image(heatmap_img, use_container_width=True)
        else:
            st.error(f"Backend error {r.status_code}: {r.text}")
