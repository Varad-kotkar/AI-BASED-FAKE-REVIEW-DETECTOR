
import sqlite3
from datetime import datetime

DB_PATH = "review_checker.db"


def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT,
            review_text TEXT,
            rating INTEGER,
            prediction TEXT,
            fake_probability REAL,
            trust_score REAL,
            created_at TEXT
        )
        """
    )
    db.commit()
    db.close()


def save_review(db, product_id, result):
    db.execute(
        """
        INSERT INTO reviews (product_id, review_text, rating, prediction, fake_probability, trust_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            product_id,
            result["review_text"],
            result["rating"],
            result["prediction"],
            result["fake_probability"],
            result["trust_score"],
            datetime.now().isoformat(),
        ),
    )
    db.commit()


def get_analytics(db):
    rows = db.execute("SELECT * FROM reviews").fetchall()
    total = len(rows)

    if total == 0:
        return {
            "total_reviews": 0,
            "fake_reviews": 0,
            "fake_percentage": 0,
            "average_trust_score": 0,
        }

    fake_count = sum(1 for row in rows if row["prediction"] == "fake")
    avg_trust = sum(row["trust_score"] for row in rows) / total

    return {
        "total_reviews": total,
        "fake_reviews": fake_count,
        "fake_percentage": round((fake_count / total) * 100, 2),
        "average_trust_score": round(avg_trust, 2),
    }
