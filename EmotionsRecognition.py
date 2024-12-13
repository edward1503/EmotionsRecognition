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
vectorizer = TfidfVectorizer(max_features=5000)


# Tải các mô hình đã huấn luyện
svm_model = joblib.load('svm_model.pkl')
nb_model = joblib.load('nb_model.pkl')

# Danh sách các nhãn cảm xúc
labels = ['Fear', 'Anger', 'Surprise', 'Enjoyment', 'Disgust', 'Sadness', 'Other']

# Hàm dự đoán cho SVM và Naive Bayes
def predict_svm_nb(model, text):
    text_vectorized = vectorizer.transform([text])  # Biến 'text' thành một vector
    prob = model.predict_proba(text_vectorized)[0]
    return prob

# Streamlit App
st.title('Emotion Prediction for Text')
st.write("Nhập một câu văn để dự đoán cảm xúc.")

# Nhập câu văn từ người dùng
user_input = st.text_area("Nhập câu văn", "Ước gì sau này về già vẫn có thể như cụ này :)")

# Lựa chọn mô hình để dự đoán
model_choice = st.selectbox('Chọn mô hình', ['SVM', 'Naive Bayes'])

# Biểu đồ xác suất
def plot_probabilities(probabilities):
    plt.figure(figsize=(10,6))
    sns.barplot(x=labels, y=probabilities, palette="viridis")
    plt.xlabel('Emotion Classes')
    plt.ylabel('Probability')
    plt.title('Predicted Emotion Probabilities')
    st.pyplot(plt)

# Dự đoán và hiển thị kết quả
if model_choice == 'SVM':
    probabilities = predict_svm_nb(svm_model, user_input)
elif model_choice == 'Naive Bayes':
    probabilities = predict_svm_nb(nb_model, user_input)


# Hiển thị kết quả dự đoán
st.write("Xác suất dự đoán cho các lớp cảm xúc:")
plot_probabilities(probabilities)

# Hiển thị nhãn có xác suất cao nhất
predicted_label = labels[np.argmax(probabilities)]
st.write(f"Nhãn dự đoán cao nhất: {predicted_label} với xác suất {probabilities[np.argmax(probabilities)]:.2f}")
