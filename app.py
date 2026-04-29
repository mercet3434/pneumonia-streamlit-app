import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

@st.cache_resource
def load_pneumonia_model():
    return load_model("pneumonia_mobilenetv2_final.keras")

model = load_pneumonia_model()

THRESHOLD = 0.70

st.title("🫁 Chest X-Ray Pneumonia Detection")
st.write(
    "Bu uygulama göğüs röntgen görüntüsünü analiz ederek "
    "**NORMAL** veya **PNEUMONIA** tahmini yapar."
)

st.info("Final model: MobileNetV2 | Threshold: 0.70 | Test Accuracy: %90.22")

uploaded_file = st.file_uploader(
    "Bir göğüs röntgen görüntüsü yükleyin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Yüklenen X-Ray Görüntüsü", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array, verbose=0)[0][0]

    if pred_prob > THRESHOLD:
        prediction = "PNEUMONIA"
        st.error(f"Tahmin: {prediction}")
    else:
        prediction = "NORMAL"
        st.success(f"Tahmin: {prediction}")

    st.write(f"**Pneumonia Skoru:** {pred_prob:.4f}")
    st.write(f"**Kullanılan Threshold:** {THRESHOLD}")

    st.progress(float(pred_prob))

    st.caption(
        "Not: Bu uygulama eğitim amaçlıdır. Gerçek tıbbi tanı için kullanılamaz."
    )
