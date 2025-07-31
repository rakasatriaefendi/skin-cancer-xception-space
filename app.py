import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import io

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

# Model dan class list (harus sesuai urutan model output)
class_keys = list(folder_to_label.keys())

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.h5", compile=False)

model = load_model()

def predict(image):
    image = image.resize((224, 224))
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)[0]
    confidence = np.max(predictions)
    index = np.argmax(predictions)
    key = class_keys[index]
    return key, confidence, predictions

def generate_pdf(info, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Hasil Klasifikasi Citra Penyakit Kulit", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Label Prediksi: {info['name']}\nKategori: {info['category']}\nConfidence: {confidence:.2%}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Deskripsi:\n{info['description']}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Penanganan:\n{info['treatment']}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Rekomendasi:\n{info['recommendation']}")
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# UI
st.title("🔬 Klasifikasi Citra Penyakit Kulit")
st.markdown("Unggah gambar kulit untuk mendapatkan prediksi penyakit berdasarkan model CNN.")

uploaded_file = st.file_uploader("📤 Unggah gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Gambar yang diunggah", use_column_width=True)

    with st.spinner("🔍 Memprediksi..."):
        key, confidence, _ = predict(image)
        info = disease_info[key]

        st.success(f"✅ Prediksi: **{info['name']}**")
        st.info(f"🧠 Tingkat Keyakinan: **{confidence:.2%}**")
        st.write(f"**Kategori:** {info['category']}")
        st.write("### 📝 Deskripsi Medis")
        st.write(info['description'])
        st.write("### 💊 Penanganan")
        st.write(info['treatment'])
        st.write("### 📌 Rekomendasi")
        st.write(info['recommendation'])

        pdf = generate_pdf(info, confidence)
        st.download_button("📄 Unduh Laporan PDF", data=pdf, file_name="hasil_prediksi.pdf", mime="application/pdf")
