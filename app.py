# ====== [IMPORT & PAGE CONFIG] ======
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

st.set_page_config(page_title="Skin Cancer Classifier", layout="wide")

# ====== [LABEL & PENJELASAN] ======
folder_to_label = {
    'akiec': ('Actinic Keratoses', 'Tidak Ganas'),
    'bcc': ('Basal Cell Carcinoma', 'Ganas'),
    'bkl': ('Benign Keratosis Like Lesions', 'Tidak Ganas'),
    'df': ('Dermatofibroma', 'Tidak Ganas'),
    'mel': ('Melanoma', 'Ganas'),
    'nv': ('Melanocytic Nevi', 'Aman'),
    'vasc': ('Vascular Lesions', 'Tidak Ganas')
}

disease_info = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'category': 'Tidak Ganas',
        'description': "Lesi kulit akibat paparan sinar matahari kronis.",
        'treatment': "Krim topikal, krioterapi, atau terapi fotodinamik.",
        'recommendation': "Disarankan konsultasi ke dokter kulit untuk evaluasi."
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'category': 'Ganas',
        'description': "Kanker kulit paling umum, tumbuh lambat dan jarang menyebar.",
        'treatment': "Eksisi, kuretase, cryotherapy.",
        'recommendation': "Segera temui dokter spesialis kulit."
    },
    'bkl': {
        'name': 'Benign Keratosis Like Lesions',
        'category': 'Tidak Ganas',
        'description': "Lesi jinak seperti solar lentigo dan seboroik keratosis.",
        'treatment': "Dapat diangkat untuk alasan kosmetik.",
        'recommendation': "Pemeriksaan rutin tetap disarankan."
    },
    'df': {
        'name': 'Dermatofibroma',
        'category': 'Tidak Ganas',
        'description': "Benjolan kecil akibat iritasi atau gigitan serangga.",
        'treatment': "Tidak memerlukan perawatan khusus.",
        'recommendation': "Konsultasi jika tumbuh atau terasa sakit."
    },
    'mel': {
        'name': 'Melanoma',
        'category': 'Ganas',
        'description': "Kanker kulit paling berbahaya.",
        'treatment': "Eksisi dan imunoterapi.",
        'recommendation': "Segera ke dokter spesialis onkologi kulit."
    },
    'nv': {
        'name': 'Melanocytic Nevi',
        'category': 'Aman',
        'description': "Tahi lalat jinak, bentuk simetris.",
        'treatment': "Tidak perlu tindakan kecuali ada perubahan mencurigakan.",
        'recommendation': "Pantau secara berkala."
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'category': 'Tidak Ganas',
        'description': "Lesi pembuluh darah seperti angioma.",
        'treatment': "Laser atau pembedahan minor jika perlu.",
        'recommendation': "Konsultasi jika membesar atau berdarah."
    }
}

# ====== [LOAD MODEL] ======
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.h5")
model = load_model()

# ====== [UI - JUDUL & UPLOAD] ======
st.title("🔬 Skin Cancer Detection from Image")
st.markdown("Upload gambar kulit untuk memprediksi jenis lesi dan mendapatkan informasi medis edukatif.")

uploaded_file = st.file_uploader("📤 Upload a skin image", type=["jpg", "jpeg", "png"])

# ====== [JIKA FILE DIUPLOAD] ======
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Gambar yang Diunggah", use_column_width=True)

    with col2:
        resized_image = image.resize((224, 224))
        img_array = np.array(resized_image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(prediction[0][pred_class])

        label = list(folder_to_label.values())[pred_class]
        result_label, result_category = label

        st.success(f"✅ Hasil Prediksi: {result_label}")
        st.write(f"📂 Kategori: **{result_category}**")
        st.progress(int(confidence * 100))
        st.markdown(f"### 🔒 Confidence: **{confidence:.2%}**")

        # ====== [PLOT KONFIDENSI] ======
        class_names = [v[0] for v in folder_to_label.values()]
        plt.figure(figsize=(10, 3))
        plt.bar(class_names, prediction[0], color='skyblue')
        plt.xticks(rotation=30)
        plt.ylabel("Confidence")
        plt.title("Confidence per Class")
        st.pyplot(plt)

        # ====== [INFO MEDIS] ======
        info = list(disease_info.values())[pred_class]
        with st.expander("🧾 Informasi Medis Lengkap"):
            st.markdown(f"### 🧬 {info['name']} ({info['category']})")
            st.write(f"**Deskripsi:** {info['description']}")
            st.write(f"**Penanganan:** {info['treatment']}")
            st.info(info['recommendation'])

        # ====== [PDF GENERATOR FIXED] ======
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Skin Lesion Classification Report", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Predicted Class: {result_label}", ln=True)
            pdf.cell(200, 10, txt=f"Category: {result_category}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)
            pdf.ln(5)
            pdf.multi_cell(0, 10, f"Description: {info['description']}")
            pdf.multi_cell(0, 10, f"Treatment: {info['treatment']}")
            pdf.multi_cell(0, 10, f"Recommendation: {info['recommendation']}")
            pdf.ln(10)
            pdf.set_text_color(128)
            pdf.set_font("Arial", "I", 10)
            pdf.multi_cell(0, 10, "Disclaimer: This is not a formal diagnosis. Always consult a medical professional.")

            # ✅ Output as bytes and wrap in BytesIO
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            return io.BytesIO(pdf_bytes)

        pdf_file = generate_pdf()
        st.download_button(
            label="📄 Download Laporan PDF",
            data=pdf_file,
            file_name="skin_lesion_report.pdf",
            mime="application/pdf"
        )
