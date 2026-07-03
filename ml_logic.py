"""Fake review checker - simple version"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = "dataset/fake reviews dataset.csv"

model = None
vectorizer = None


# ---------------------------
# Step 1: Data Preprocessing
# ---------------------------

def clean_text(text):
    text = str(text).lower()                              # lowercase
    text = re.sub(r"http\S+", "", text)                    # remove links
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", "", text)                        # remove numbers
    text = re.sub(r"\s+", " ", text).strip()                # remove extra spaces
    return text


def load_data():
    data = pd.read_csv(DATASET_PATH)
    data = data.rename(columns={"text_": "text"})
    data = data.dropna(subset=["text", "label"])

    # convert labels to 0 and 1
    data["label"] = data["label"].map({"CG": 1, "OR": 0})

    # clean the review text
    data["clean_text"] = data["text"].apply(clean_text)

    # remove empty rows after cleaning
    data = data[data["clean_text"] != ""]

    return data


# ---------------------------
# Step 2: Train Model
# ---------------------------

def train_model():
    global model, vectorizer

    data = load_data()

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vectorizer.fit_transform(data["clean_text"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Model trained. Accuracy:", accuracy)


# ---------------------------
# Step 3: Predict Review
# ---------------------------

def predict_review(review_text, rating=3):
    cleaned_text = clean_text(review_text)
    vector = vectorizer.transform([cleaned_text])
    fake_probability = model.predict_proba(vector)[0][1]

    # simple extra check for suspicious/promotional words
    suspicious_words = ["amazing", "awesome", "best", "excellent", "perfect", "must buy"]
    found_words = [word for word in suspicious_words if word in cleaned_text]
    if found_words:
        fake_probability += 0.1

    if fake_probability > 1:
        fake_probability = 1.0

    prediction = "fake" if fake_probability >= 0.5 else "real"
    trust_score = round((1 - fake_probability) * 100, 2)

    return {
        "prediction": prediction,
        "fake_probability": round(fake_probability, 2),
        "trust_score": trust_score,
        "suspicious_words": found_words,
        "review_text": review_text,
        "rating": rating,
    }


# ---------------------------
# Step 4: Summarize Reviews
# ---------------------------

def summarize_product_reviews(reviews, ratings=None):
    ratings = ratings or []
    results = []

    for i, review in enumerate(reviews):
        rating = ratings[i] if i < len(ratings) else 3
        results.append(predict_review(review, rating))

    total = len(results)
    if total == 0:
        return {"message": "No reviews found"}

    fake_count = sum(1 for r in results if r["prediction"] == "fake")
    fake_percentage = round((fake_count / total) * 100, 2)
    avg_trust = round(sum(r["trust_score"] for r in results) / total, 2)

    if avg_trust >= 75:
        recommendation = "Buy"
    elif avg_trust >= 55:
        recommendation = "Caution"
    else:
        recommendation = "Avoid"

    return {
        "total_reviews": total,
        "fake_review_percentage": fake_percentage,
        "average_trust_score": avg_trust,
        "buy_recommendation": recommendation,
        "review_results": results,
    }


# train the model once when this file runs
train_model()
