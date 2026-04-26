"""Simple ML, refund, and review scraping logic for the project."""

import base64
import binascii
import io
import json
import re
import string
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageStat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset" / "fake reviews dataset.csv"
FALLBACK_DATASET_PATH = PROJECT_DIR / "dataset" / "sentiment.csv"
MODELS_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

SUSPICIOUS_WORDS = [
    "amazing",
    "awesome",
    "best",
    "excellent",
    "perfect",
    "super",
    "unbelievable",
    "must buy",
    "guaranteed",
]

model = None
vectorizer = None


def clean_text(text):
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_training_data():
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        df = df.rename(columns={"text_": "text"})
        df = df[["text", "rating", "label"]].dropna()
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).clip(1, 5)
        df["label"] = df["label"].map({"CG": 1, "OR": 0}).fillna(0).astype(int)
    else:
        df = pd.read_csv(FALLBACK_DATASET_PATH, encoding="latin1", low_memory=False)
        df["text"] = (df["Summary"].fillna("") + " " + df["Review"].fillna("")).str.strip()
        df["rating"] = pd.to_numeric(df["Rate"], errors="coerce").fillna(3).clip(1, 5)
        df["label"] = df["Sentiment"].apply(lambda value: 1 if value == "negative" else 0)

    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"] != ""].drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    return df


def train_and_save_model():
    df = load_training_data()

    simple_vectorizer = TfidfVectorizer(max_features=4000, min_df=2, stop_words="english")
    X = simple_vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    simple_model = LogisticRegression(max_iter=500, class_weight="balanced")
    simple_model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, simple_model.predict(X_test))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(simple_model, MODEL_PATH)
    joblib.dump(simple_vectorizer, VECTORIZER_PATH)

    metadata = {
        "model": "Logistic Regression",
        "dataset": str(DATASET_PATH if DATASET_PATH.exists() else FALLBACK_DATASET_PATH),
        "rows": int(len(df)),
        "accuracy": round(float(accuracy), 4),
        "fake_label": "CG",
        "real_label": "OR",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return simple_model, simple_vectorizer, metadata


def load_or_train_model():
    global model, vectorizer

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        test_vector = vectorizer.transform(["test review"])
        if getattr(model, "n_features_in_", test_vector.shape[1]) != test_vector.shape[1]:
            raise ValueError("Old model shape does not match current vectorizer.")
    except Exception:
        model, vectorizer, _metadata = train_and_save_model()

    return model, vectorizer


def find_suspicious_words(cleaned_text):
    found_words = []
    for word in SUSPICIOUS_WORDS:
        if word in cleaned_text:
            found_words.append({"word": word, "reason": "too promotional"})
    return found_words


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def predict_review(review_text, rating=3):
    global model, vectorizer

    if model is None or vectorizer is None:
        load_or_train_model()

    cleaned_text = clean_text(review_text)
    text_vector = vectorizer.transform([cleaned_text])
    model_score = float(model.predict_proba(text_vector)[0][1])

    words = cleaned_text.split()
    suspicious_words = find_suspicious_words(cleaned_text)
    rating_value = int(rating or 3)

    rule_score = 0.0
    if len(words) < 8:
        rule_score += 0.10
    if rating_value in (1, 5):
        rule_score += 0.05
    if str(review_text).count("!") >= 2:
        rule_score += 0.10
    if suspicious_words:
        rule_score += min(0.15, len(suspicious_words) * 0.04)

    fake_probability = clamp((model_score * 0.85) + rule_score)
    prediction = "fake" if fake_probability >= 0.5 else "real"
    trust_score = round((1 - fake_probability) * 100, 2)

    if prediction == "fake":
        explanation = "This review has text patterns that look similar to generated or copied reviews."
    else:
        explanation = "This review looks closer to normal customer writing."

    return {
        "prediction": prediction,
        "probability_score": round(fake_probability, 4),
        "fake_probability": round(fake_probability, 4),
        "text_pattern_score": round(model_score, 4),
        "similarity_score": 0.0,
        "trust_score": trust_score,
        "explanation": explanation,
        "suspicious_words": suspicious_words,
        "model_name": "logistic_regression",
        "clean_text": cleaned_text,
    }


def summarize_product_reviews(reviews, ratings=None, product_url=None, product_id=None):
    ratings = ratings or []
    review_results = []

    for index, review in enumerate(reviews):
        rating = ratings[index] if index < len(ratings) else 3
        result = predict_review(review, rating)
        result["review_text"] = review
        result["rating"] = int(rating)
        review_results.append(result)

    if not review_results:
        return {
            "product_url": product_url,
            "product_id": product_id,
            "product_trust_score": 0.0,
            "fake_review_percentage": 0.0,
            "buy_recommendation": "insufficient data",
            "summary": "No reviews were available for analysis.",
            "review_results": [],
            "review_volume": 0,
        }

    total = len(review_results)
    fake_count = sum(1 for item in review_results if item["prediction"] == "fake")
    fake_percentage = round((fake_count / total) * 100, 2)
    trust_score = round(sum(item["trust_score"] for item in review_results) / total, 2)

    if trust_score >= 75 and fake_percentage < 25:
        recommendation = "Buy"
    elif trust_score >= 55 and fake_percentage < 45:
        recommendation = "Caution"
    else:
        recommendation = "Avoid"

    return {
        "product_url": product_url,
        "product_id": product_id,
        "product_trust_score": trust_score,
        "fake_review_percentage": fake_percentage,
        "buy_recommendation": recommendation,
        "summary": f"Analyzed {total} reviews. Average trust score is {trust_score}.",
        "review_results": review_results,
        "review_volume": total,
    }


def decode_image(image_payload):
    if not image_payload:
        return None

    payload = image_payload.split(",", 1)[1] if "," in image_payload else image_payload
    try:
        image_bytes = base64.b64decode(payload)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (binascii.Error, ValueError, OSError):
        return None


def average_hash(image, size=8):
    small_image = image.resize((size, size)).convert("L")
    pixels = list(small_image.getdata())
    average = sum(pixels) / len(pixels)
    return "".join("1" if pixel > average else "0" for pixel in pixels)


def hash_similarity(hash_a, hash_b):
    if not hash_a or not hash_b or len(hash_a) != len(hash_b):
        return 0.0
    differences = sum(left != right for left, right in zip(hash_a, hash_b))
    return 1 - (differences / len(hash_a))


def image_variation_score(image):
    gray_image = image.convert("L")
    standard_deviation = ImageStat.Stat(gray_image).stddev[0]
    return clamp(standard_deviation / 100)


def analyze_refund_fraud(product_name, damage_image, refund_amount, existing_hashes=None):
    image = decode_image(damage_image)
    existing_hashes = existing_hashes or []

    if image is None:
        image_hash = None
        image_similarity = 0.0
        image_pattern = 0.25
        product_match = "no evidence image supplied"
    else:
        image_hash = average_hash(image)
        image_similarity = max([hash_similarity(image_hash, old_hash) for old_hash in existing_hashes] or [0.0])
        image_pattern = clamp((image_similarity * 0.70) + ((1 - image_variation_score(image)) * 0.30))
        product_match = "image received for manual product match"

    amount_risk = clamp(float(refund_amount or 0) / 5000)
    fraud_score = clamp((image_similarity * 0.45) + (image_pattern * 0.35) + (amount_risk * 0.20))

    if fraud_score >= 0.70:
        risk_level = "high"
        recommendation = "Escalate this refund for manual review."
    elif fraud_score >= 0.45:
        risk_level = "medium"
        recommendation = "Ask for more proof before approving the refund."
    else:
        risk_level = "low"
        recommendation = "The claim looks low risk based on the current details."

    return {
        "product_name": product_name,
        "image_pattern_score": round(image_pattern, 4),
        "image_similarity_score": round(image_similarity, 4),
        "product_match_result": product_match,
        "fraud_score": round(fraud_score, 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "image_hash": image_hash,
        "details": {
            "refund_amount_risk": round(amount_risk, 4),
        },
    }


def clean_review_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def get_page_title(soup):
    title = soup.select_one("h1") or soup.select_one("title")
    return clean_review_text(title.get_text(" ", strip=True)) if title else ""


def extract_reviews_from_url(product_url, max_reviews=10):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(product_url, timeout=20, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    selectors = [
        '[data-hook="review-body"]',
        '[itemprop="reviewBody"]',
        ".review-body",
        ".review-content",
        ".review-text",
        ".review",
    ]

    reviews = []
    seen = set()
    for selector in selectors:
        for node in soup.select(selector):
            text = clean_review_text(node.get_text(" ", strip=True))
            key = text.lower()
            if len(text) >= 25 and key not in seen:
                reviews.append(text)
                seen.add(key)
            if len(reviews) >= max_reviews:
                break
        if len(reviews) >= max_reviews:
            break

    hostname = urlparse(product_url).netloc or "this page"
    if not reviews:
        return {
            "product_url": product_url,
            "product_title": get_page_title(soup),
            "reviews": [],
            "review_count": 0,
            "status": "unsupported",
            "message": f"Reviews were not found on {hostname}.",
        }

    return {
        "product_url": product_url,
        "product_title": get_page_title(soup),
        "reviews": reviews,
        "review_count": len(reviews),
        "status": "success",
        "message": "Reviews were extracted successfully.",
    }
