# 📧 Email Spam Detector (Streamlit + Neural Network)

This is a web-based **Email Spam Classifier** built with **Streamlit** and a **Neural Network** (TensorFlow). It allows users to input the content of an email and determine whether it's **Spam** or **Not Spam (Ham)**.

## 🚀 Features

- 🔍 Uses **TF-IDF vectorization** to transform email text into numeric features
- 🤖 A **Neural Network** built with TensorFlow for classification
- 📊 Displays prediction confidence and model accuracy
- 🧠 Shows most common words found in spam and ham emails
- 🕒 Displays last training date and accuracy score

## 🧠 Model Architecture
This uses simple feedforward neural network
- Input: TF-IDF vector of 5000 features
- Hidden Layer 1: Dense(128) + ReLU + Dropout(0.5)
- Hidden Layer 2: Dense(64) + ReLU + Dropout(0.5)
- Output: Dense(1) + Sigmoid (binary classification)

## 🗃️ Dataset

The model is trained using a labeled dataset of spam and ham emails (`spamham.csv`). The emails are cleaned and stopwords are removed before feature extraction.



# 🛠️ How to Use

### 1. Install dependencies:

```
pip install -r requirements.txt
```

### 2.Train the Model:

```
python train_model.py
```

### 3. Run the streamlit app:

```
streamlit run app/spam_app.py
```

📌 Notes
The neural network and vectorizer are saved and reused between sessions.

Includes training insights like top 20 spam/ham words.

Designed as a demo. For production use, retrain with domain-specific data.


## 📂 Dataset Source

The dataset `spamham.csv` used in this project was sourced from [Balakishan77's Spam Email Classifier Repository](https://github.com/Balakishan77/Spam-Email-Classifier).  
All credit for the dataset goes to the original author.
