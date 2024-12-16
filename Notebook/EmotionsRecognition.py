import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import joblib
import re
import emoji
from pyvi import ViTokenizer
import matplotlib.pyplot as plt
import tensorflow as tf
emotions = ['Surprise', 'Sadness', 'Anger', 'Fear', 'Enjoyment', 'Disgust', 'Other']
emotion_to_num = dict({v:k for k,v in enumerate(emotions)})
num_to_emotion = dict({k:v for k,v in enumerate(emotions)})

# Load teencode
teencodes = {}
with open(r'D:\VSCODE\EmotionsRecognition\Data\teencode4.txt','r', encoding="utf8") as file:
    file = file.read()
    lines = file.split('\n')
    for line in lines:
        elements = line.split('\t')
        if len(elements) == 2:
            teencodes[elements[0]] = elements[1]
def preprocess_teencodes(sentence):
    for key, value in teencodes.items():
        sentence = re.sub(r'\b{}\b'.format(key), value, sentence)
    return sentence

# Load stopwords
stopwords = []
with open(r'D:\VSCODE\EmotionsRecognition\Data\vietnamese-stopwords.txt', 'r', encoding="utf8") as f:
    for line in f:
        stopwords.append(line.strip())
def remove_stopwords(words):
    new_words = [word for word in words if re.sub("_", " ", word) not in stopwords]
    return new_words

# Load Emoji
def preprocess_emoji(sentence):
    emotion_dict = {
        '(:|;|=)+(\)|\]|>)+':'🙂','(:|;|=)+(\(|\[|<)+':'😞','(:|;|=)+(D|d)':'😁',
        '(-_-)|(-\.-)':'😐',':v':'_pacman_smile_','(:|;|=)+(\'|`|\")+(\)|\]|>)+':'🥲','(:|;|=)+(\'|`|\")+(\(|\[|<)+':'😢',
        '@@':'😵‍💫','đc':'được','đk':'được','bik':'biết','ngừi':'người','hix':'hic','lm':'làm'
    }
    for key, value in emotion_dict.items():
        sentence = re.sub(key,value,sentence)
    sentence = emoji.demojize(sentence)
    sentence = re.sub(r":(.*?):",r" _\1_ ",sentence)
    sentence = re.sub(r'([!@#$%^&*()_+={}:;"\'<>,?/\|~-])\1+',r'\1',sentence)
    return sentence

# Tokenize
def tokenize(sentence):
    start_token = ' _s_ '
    end_token = ' _e_ '
    sentence = sentence.lower()

    # Preprocess Emoji
    sentence = preprocess_emoji(sentence)

    # Preprocess teencodes
    sentence = preprocess_teencodes(sentence)

    sentence = start_token + sentence + end_token
    return ViTokenizer.tokenize(sentence).split()
def get_sentence_vector(sentence):
    word_vectors = [w2v.wv[word] for word in sentence if word in w2v.wv]
    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:
        return [0] * w2v.vector_size
@st.cache_resource
def load_models():
    # Load các vectorizer và model
    tfidf = joblib.load(r'D:\VSCODE\EmotionsRecognition\model\tfidf_vectorizer.pkl')
    w2v = joblib.load(r'D:\VSCODE\EmotionsRecognition\model\word2vec_model.pkl')

    # TF-IDF models
    maxent_tfidf = joblib.load(r'D:\VSCODE\EmotionsRecognition\model\maxent_tfidf.pkl')
    svm_tfidf = joblib.load(r'D:\VSCODE\EmotionsRecognition\model\svm_tfidf.pkl')
    logistic_tfidf = joblib.load(r'D:\VSCODE\EmotionsRecognition\model\logistic_tfidf.pkl')
    nn_tfidf = tf.keras.models.load_model(r'D:\VSCODE\EmotionsRecognition\model\lstm_tfidf_ct_model.h5')
    # W2V models
    maxent_w2v = joblib.load(r'D:\VSCODE\EmotionsRecognition\model\maxent_w2v.pkl')
    svm_w2v = joblib.load(r'D:\VSCODE\EmotionsRecognition\model\svm_w2v.pkl')
    logistic_w2v = joblib.load(r'D:\VSCODE\EmotionsRecognition\model\logistic_w2v.pkl')
    bilstm_w2v = tf.keras.models.load_model(r'D:\VSCODE\EmotionsRecognition\model\best_bilstm_w2v_model.h5')
    return tfidf, w2v, maxent_tfidf, svm_tfidf, logistic_tfidf, nn_tfidf, maxent_w2v, svm_w2v, logistic_w2v, bilstm_w2v

# Gọi hàm load_models() chỉ một lần
(tfidf, w2v, maxent_tfidf, svm_tfidf, logistic_tfidf, nn_tfidf, 
 maxent_w2v, svm_w2v, logistic_w2v, bilstm_w2v) = load_models()
max_len = 50
word_index = {word: index for index, word in enumerate(w2v.wv.index_to_key, start=1)}
def sentence_to_sequence(sentence, word_index, max_len):
    sequence = [word_index.get(word, 0) for word in sentence]  # 0 nếu từ không có trong Word2Vec
    return sequence[:max_len] + [0] * max(0, max_len - len(sequence))  # Padding

# Title
st.title("Emotion Recognition 😊😍😂😢")
# Heading
st.write("This is a simple Emotion Recognition web app to predict the emotion of a Vietnamese text.")
# Text area
user_input = st.text_area("Enter your text here:", height=100)
col1, col2 = st.columns(2)

# Radio button để chọn vectorizer
with col1:
    vectorizer = st.radio("Choose a vectorizer:", ("TF-IDF", "Word2Vec"))

# Radio button để chọn mô hình
with col2:
    model = st.radio("Choose a model:", ("SVM", "MaxEnt", "Logistic Regression", "BiLSTM", "Normal NN"))
def validate_input(text):
    # Kiểm tra nếu văn bản trống hoặc chỉ chứa khoảng trắng
    if not text.strip():
        return False
    return True
if not validate_input(user_input):
    st.error("Please enter some text. The input cannot be empty or just spaces.")
else:
    # Sử dụng 2 cột để sắp xếp các radio button

    text = user_input
    text = tokenize(text)
    text = remove_stopwords(text)
    # Button to predict
    if st.button("Predict"):
        if vectorizer == "TF-IDF":
            text = tfidf.transform([' '.join(text)])
            if model == "SVM":
                prediction = svm_tfidf.predict(text)
                probabilities = svm_tfidf.predict_proba(text)
            elif model == "MaxEnt":
                prediction = maxent_tfidf.classify({'features': tuple(text.toarray()[0].tolist())})
            elif model == "Normal NN":
                text = text.toarray()
                prediction = np.argmax(nn_tfidf.predict(text), axis=1)
                probabilities = nn_tfidf.predict(text)
            else:
                prediction = logistic_tfidf.predict(text)
                probabilities = logistic_tfidf.predict_proba(text)
        else:
            if model == "SVM":
                text = get_sentence_vector(text)
                prediction = svm_w2v.predict([text])
                probabilities = svm_w2v.predict_proba([text])
            elif model == "MaxEnt":
                #text = text.values.reshape(1, -1)
                text = get_sentence_vector(text)
                prediction = maxent_w2v.classify({'features': tuple(text.tolist())})
            elif model == "BiLSTM":
                text = np.array([sentence_to_sequence(text, word_index, max_len)])
                probabilities = bilstm_w2v.predict(text)
                prediction = np.argmax(probabilities, axis=1)
            else:
                text = get_sentence_vector(text)
                prediction = logistic_w2v.predict([text])
                probabilities = logistic_w2v.predict_proba([text])
        st.write("Prediction:", num_to_emotion[prediction] if model == "MaxEnt" else num_to_emotion[prediction[0]])
        if model != "MaxEnt":
            emotion_probabilities = probabilities[0]  # Xác suất của các nhãn
            # Vẽ biểu đồ xác suất
            fig, ax = plt.subplots()
            ax.bar(num_to_emotion.values(), emotion_probabilities, color='skyblue')
            ax.set_xlabel('Emotion')
            ax.set_ylabel('Probability')
            ax.set_title('Emotion Probabilities')
            st.pyplot(fig)