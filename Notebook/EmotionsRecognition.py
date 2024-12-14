import re
import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Hàm tiền xử lý văn bản
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fa5\u3040-\u30ff\u3130-\u318f\uAC00-\uD7A3\w\s\u2600-\u26FF\u2700-\u27BF\u1F600-\u1F64F\u1F300-\u1F5FF]', '', text)  # Giữ lại emoji và các ký tự đặc biệt khác
    text = re.sub(r'\s+', ' ', text).strip()  # Xóa khoảng trắng thừa
    return text

# Tiền xử lý và vector hóa
vectorizer = joblib.load('./Model/tfidf_vectorizer.pkl')


# Tải các mô hình đã huấn luyện
svm_model = joblib.load('./Model/svm_model.pkl')
nb_model = joblib.load('./Model/nb_model.pkl')

# Danh sách các nhãn cảm xúc
labels = ['Fear', 'Anger', 'Surprise', 'Enjoyment', 'Disgust', 'Sadness', 'Other']

# Hàm dự đoán cho SVM và Naive Bayes
def predict_svm_nb(model, text, vectorizer):
    if vectorizer is None:
        raise ValueError("Vectorizer is not loaded properly")
    
    # Mã hóa văn bản đầu vào
    text_vectorized = vectorizer.transform([text])  # Biến 'text' thành một vector
    prob = model.predict_proba(text_vectorized)[0]  # Dự đoán xác suất
    return prob

# Streamlit App
st.title('Emotion Prediction for Text')
st.write("Nhập một câu văn để dự đoán cảm xúc.")

# Giao diện người dùng
user_input = st.text_input("Nhập câu văn để dự đoán cảm xúc:")
model_choice = st.selectbox("Chọn mô hình để dự đoán:", ["Naive Bayes", "SVM"])

if user_input:
    if model_choice == "Naive Bayes":
        probabilities = predict_svm_nb(nb_model, user_input, vectorizer)
    elif model_choice == "SVM":
        probabilities = predict_svm_nb(svm_model, user_input, vectorizer)
    # Hiển thị kết quả nhãn dự đoán
    predicted_label = labels[np.argmax(probabilities)]  # Nhãn với xác suất cao nhất
    st.subheader(f"Kết quả dự đoán: {predicted_label}")

    # Hiển thị xác suất cho từng nhãn
    st.write("Xác suất của các nhãn cảm xúc:")
    for i, prob in enumerate(probabilities):
        st.write(f"{labels[i]}: {prob:.4f}")
    
    # Vẽ biểu đồ phân bố xác suất
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, probabilities, color='skyblue')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Probability')
    ax.set_title('Probability Distribution of Emotions')
    plt.xticks(rotation=45)
    st.pyplot(fig)
