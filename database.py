"""Very small SQLite database file used by the FastAPI app."""

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "review_checker.db"


def open_database():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    db = open_database()

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS review_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT,
            reviewer_id TEXT,
            review_text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            text_pattern_score REAL NOT NULL,
            sentiment_mismatch REAL NOT NULL,
            similarity_score REAL NOT NULL,
            trust_score REAL NOT NULL,
            explanation TEXT NOT NULL,
            suspicious_words TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS refund_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            product_id TEXT,
            order_id TEXT,
            claim_reason TEXT,
            refund_amount REAL,
            image_hash TEXT,
            image_pattern_score REAL NOT NULL,
            image_similarity_score REAL NOT NULL,
            product_match_result TEXT NOT NULL,
            fraud_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS reviewer_profiles (
            reviewer_id TEXT PRIMARY KEY,
            total_reviews INTEGER NOT NULL,
            avg_rating REAL NOT NULL,
            suspicious_score REAL NOT NULL,
            last_review_at TEXT NOT NULL
        )
        """
    )

    db.commit()
    db.close()


def get_db():
    db = open_database()
    try:
        yield db
    finally:
        db.close()


def now_text():
    return datetime.utcnow().isoformat(sep=" ", timespec="seconds")


def store_review_prediction(db, payload):
    values = dict(payload)
    values["created_at"] = now_text()
    values["suspicious_words"] = json.dumps(values.get("suspicious_words") or [])

    columns = [
        "product_id",
        "reviewer_id",
        "review_text",
        "rating",
        "prediction",
        "fake_probability",
        "text_pattern_score",
        "sentiment_mismatch",
        "similarity_score",
        "trust_score",
        "explanation",
        "suspicious_words",
        "created_at",
    ]
    placeholders = ", ".join("?" for _ in columns)
    cursor = db.execute(
        f"INSERT INTO review_predictions ({', '.join(columns)}) VALUES ({placeholders})",
        [values.get(column) for column in columns],
    )
    db.commit()
    return SimpleNamespace(id=cursor.lastrowid)


def store_refund_check(db, payload):
    values = dict(payload)
    values["created_at"] = now_text()

    columns = [
        "product_name",
        "product_id",
        "order_id",
        "claim_reason",
        "refund_amount",
        "image_hash",
        "image_pattern_score",
        "image_similarity_score",
        "product_match_result",
        "fraud_score",
        "risk_level",
        "created_at",
    ]
    placeholders = ", ".join("?" for _ in columns)
    cursor = db.execute(
        f"INSERT INTO refund_checks ({', '.join(columns)}) VALUES ({placeholders})",
        [values.get(column) for column in columns],
    )
    db.commit()
    return SimpleNamespace(id=cursor.lastrowid)


def upsert_reviewer_profile(db, reviewer_id, rating, suspicious_score):
    if not reviewer_id:
        return

    row = db.execute(
        "SELECT * FROM reviewer_profiles WHERE reviewer_id = ?",
        (reviewer_id,),
    ).fetchone()

    if row is None:
        db.execute(
            """
            INSERT INTO reviewer_profiles
            (reviewer_id, total_reviews, avg_rating, suspicious_score, last_review_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (reviewer_id, 1, float(rating), float(suspicious_score), now_text()),
        )
    else:
        total_reviews = int(row["total_reviews"]) + 1
        avg_rating = ((float(row["avg_rating"]) * int(row["total_reviews"])) + float(rating)) / total_reviews
        avg_score = (
            (float(row["suspicious_score"]) * int(row["total_reviews"])) + float(suspicious_score)
        ) / total_reviews

        db.execute(
            """
            UPDATE reviewer_profiles
            SET total_reviews = ?, avg_rating = ?, suspicious_score = ?, last_review_at = ?
            WHERE reviewer_id = ?
            """,
            (total_reviews, avg_rating, avg_score, now_text(), reviewer_id),
        )

    db.commit()


def get_refund_hashes(db):
    rows = db.execute(
        "SELECT image_hash FROM refund_checks WHERE image_hash IS NOT NULL"
    ).fetchall()
    return [row["image_hash"] for row in rows if row["image_hash"]]


def get_product_trust_summary(db, product_id):
    rows = db.execute(
        "SELECT * FROM review_predictions WHERE product_id = ?",
        (product_id,),
    ).fetchall()

    if not rows:
        return {
            "product_id": product_id,
            "product_trust_score": 0.0,
            "fake_review_percentage": 0.0,
            "buy_recommendation": "insufficient data",
            "summary": "No saved reviews were found for this product.",
            "review_volume": 0,
        }

    total = len(rows)
    fake_count = sum(1 for row in rows if float(row["fake_probability"]) >= 0.5)
    trust_score = sum(float(row["trust_score"]) for row in rows) / total
    fake_percentage = (fake_count / total) * 100

    if trust_score >= 75 and fake_percentage < 25:
        recommendation = "Buy"
    elif trust_score >= 55 and fake_percentage < 45:
        recommendation = "Caution"
    else:
        recommendation = "Avoid"

    return {
        "product_id": product_id,
        "product_trust_score": round(trust_score, 2),
        "fake_review_percentage": round(fake_percentage, 2),
        "buy_recommendation": recommendation,
        "summary": f"{total} saved reviews checked. Average trust score is {round(trust_score, 2)}.",
        "review_volume": total,
    }


def get_reviewer_behavior(db, limit=8):
    rows = db.execute(
        """
        SELECT * FROM reviewer_profiles
        ORDER BY suspicious_score DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    return [
        {
            "reviewer_id": row["reviewer_id"],
            "total_reviews": row["total_reviews"],
            "avg_rating": round(float(row["avg_rating"]), 2),
            "suspicious_score": round(float(row["suspicious_score"]) * 100, 2),
        }
        for row in rows
    ]


def day_name(value):
    try:
        return datetime.fromisoformat(str(value)).strftime("%b %d")
    except ValueError:
        return str(value)[:10]


def get_analytics_summary(db, days=30):
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat(sep=" ", timespec="seconds")

    reviews = db.execute(
        "SELECT * FROM review_predictions WHERE created_at >= ? ORDER BY created_at",
        (cutoff,),
    ).fetchall()
    refunds = db.execute(
        "SELECT * FROM refund_checks WHERE created_at >= ? ORDER BY created_at",
        (cutoff,),
    ).fetchall()

    total_reviews = len(reviews)
    fake_reviews = [row for row in reviews if float(row["fake_probability"]) >= 0.5]
    fake_percent = (len(fake_reviews) / total_reviews) * 100 if total_reviews else 0
    avg_trust = sum(float(row["trust_score"]) for row in reviews) / total_reviews if total_reviews else 0
    top_reviewers = get_reviewer_behavior(db)
    top_score = max([row["suspicious_score"] for row in top_reviewers] or [0])

    review_days = Counter(day_name(row["created_at"]) for row in reviews)
    fake_days = Counter(day_name(row["created_at"]) for row in fake_reviews)
    ratings = Counter(str(row["rating"]) for row in reviews)

    return {
        "overview": {
            "totalReviews": total_reviews,
            "fakeReviewsDetected": len(fake_reviews),
            "refundFraudChecks": len(refunds),
            "highRiskRefunds": sum(1 for row in refunds if float(row["fraud_score"]) >= 0.65),
            "averageTrustScore": round(avg_trust, 2),
            "fakeReviewPercentage": round(fake_percent, 2),
            "suspiciousReviewerScore": round(top_score, 2),
            "productTrustScore": round(avg_trust, 2),
        },
        "barChart": [
            {"rating": rating, "count": count}
            for rating, count in sorted(ratings.items())
        ],
        "lineChart": [
            {"day": day, "reviews": review_days[day], "fakeReviews": fake_days[day]}
            for day in review_days
        ],
        "pieChart": [
            {"name": "Fake Reviews", "value": len(fake_reviews)},
            {"name": "Trusted Reviews", "value": max(total_reviews - len(fake_reviews), 0)},
        ],
        "topReviewers": top_reviewers,
    }
