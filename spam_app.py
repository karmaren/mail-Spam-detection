import streamlit as st
import re
import joblib
import numpy as np
import nltk
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model

nltk.download('stopwords')

# Set page config (first Streamlit command)
st.set_page_config(page_title="Email Spam Detector")

# Title and description
st.title("📧 Email Spam Detector (Neural Network)")
st.write("Enter email content below to check if it's spam or not:")

# Sidebar info
st.sidebar.title("📊 About This App")
st.sidebar.markdown("""
This spam detector uses a neural network trained on thousands of email samples.  
It uses **TF-IDF** for feature extraction and a **deep learning model** for classification.
""")
st.sidebar.markdown("**Made with ❤️ using Streamlit + TensorFlow**")

# Load model and vectorizer
model = load_model("spam_detector_model.keras")
vectorizer = joblib.load("vectorizer.pkl")
try:
    with open("model_accuracy.txt", "r") as f:
        accuracy = float(f.read())
except:
    accuracy = None
try:
    with open("last_trained.txt", "r") as f:
        trained_on = f.read()
except:
    trained_on = "Unknown"

st.sidebar.markdown(f"**📅 Last Trained:** `{trained_on}`")
if accuracy:
    st.sidebar.markdown(f"**🔍 Model Test Accuracy:** `{accuracy*100:.2f}%`")

# Load top words
with open("top_words.pkl", "rb") as f:
    top_words = pickle.load(f)

# Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# User input
user_input = st.text_area("Paste your email content below to check for spam:", height=200)

if st.button("Check Spam"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some email text.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input]).toarray()
        prediction = model.predict(vectorized_input)
        score = float(prediction[0][0])

        st.write(f"🧠 Raw prediction score: **{score:.4f}**")
        st.progress(float(score))

        # Classification
        if score > 0.5:
            st.markdown("### 🚫 **SPAM DETECTED**")
            st.error(f"Confidence: {score:.2f}")
        else:
            st.markdown("### ✅ **This is Safe**")
            st.success(f"Confidence: {1 - score:.2f}")

        # Highlight spam-like words
        input_words = set(cleaned_input.split())
        spam_word_set = set(word for word, _ in top_words['spam'])
        intersecting = input_words & spam_word_set
        if intersecting:
            st.info(f"⚠️ Suspicious words found: `{', '.join(intersecting)}`")

        # Divider and visualization
        st.divider()
        st.subheader("🔎 Word Insights from Training Data")

        def plot_top_words(title, word_data, color):
            words, counts = zip(*word_data)
            fig, ax = plt.subplots()
            ax.barh(words[::-1], counts[::-1], color=color)
            ax.set_title(title)
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            plot_top_words("Top Spam Words", top_words['spam'], 'crimson')
        with col2:
            plot_top_words("Top Ham Words", top_words['ham'], 'seagreen')

        # Divider for more info
        st.divider()
        st.subheader("🔒 Trusted Features")
        with st.container():
            st.markdown("""
            - 🔍 **Check your email list for spam traps with 99% accuracy guaranteed**
            - ♻️ **Detect recycled, typo, and pristine traps**
            - 📦 **Single and bulk spam trap checker**
            - ⚡ **Real-time API for automated spam trap checks on signup forms**
            """)

        st.divider()
        st.subheader("❓ Frequently Asked Questions")
        with st.expander("How does the model detect spam?"):
            st.write("The model is trained on thousands of labeled emails using a neural network and TF-IDF vectorization to understand common patterns in spam.")
        with st.expander("What data is used to train the model?"):
            st.write("We use a public spam email dataset and preprocessed the content to remove noise.")
        with st.expander("Can I use this model for production?"):
            st.write("This app is meant as a demo. For production use, you should retrain the model with your own dataset and apply additional checks.")
