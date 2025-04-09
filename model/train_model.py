import pandas as pd
import numpy as np
import re
import nltk
import joblib
import pickle
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from datetime import datetime
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)
os.makedirs("app", exist_ok=True)

with open("model/last_trained.txt", "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spamham.csv")
df.rename(columns={'spam': 'label'}, inplace=True)
df.dropna(subset=['label', 'text'], inplace=True)
df['label'] = df['label'].astype(int)

# Text cleaning
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

df['cleaned'] = df['text'].apply(clean_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

with open("model/model_accuracy.txt", "w") as f:
    f.write(str(accuracy))

# Save model and vectorizer
model.save("app/spam_detector_model.keras")
joblib.dump(vectorizer, "app/vectorizer.pkl")

# Save top words for insights
spam_words = " ".join(df[df['label'] == 1]['cleaned']).split()
ham_words = " ".join(df[df['label'] == 0]['cleaned']).split()
top_spam = Counter(spam_words).most_common(20)
top_ham = Counter(ham_words).most_common(20)

with open("model/top_words.pkl", "wb") as f:
    pickle.dump({'spam': top_spam, 'ham': top_ham}, f)
