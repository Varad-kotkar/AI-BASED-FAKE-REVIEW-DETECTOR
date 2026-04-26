"""Simple FastAPI application for Review Checker - All routes in one place."""

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db, init_db
from ml_logic import (
    analyze_refund_fraud,
    clean_review_text,
    extract_reviews_from_url,
    load_or_train_model,
    predict_review,
    summarize_product_reviews,
)


# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_VERSION = "3.0.0"

logger = logging.getLogger("review_checker")
logging.basicConfig(level=logging.INFO)

# Initialize database
db_ready = False
try:
    init_db()
    db_ready = True
except Exception as exc:
    logger.warning("Database initialization failed: %s", exc)

# Load ML model
load_or_train_model()

# ============================================================================
# Request and Response Models (Pydantic Schemas)
# ============================================================================


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    dataset_path: str
    model_ready: bool
    database_ready: bool


class ReviewAnalysisRequest(BaseModel):
    review_text: str = Field(..., min_length=5)
    rating: int = Field(..., ge=1, le=5)
    product_id: str | None = "product-demo"
    reviewer_id: str | None = None
    review_image: str | None = None


class ProductReviewExtractionRequest(BaseModel):
    product_url: str = Field(..., min_length=8)
    max_reviews: int = Field(default=10, ge=1, le=50)


class AnalyzeProductReviewsRequest(BaseModel):
    product_url: str = Field(..., min_length=8)
    reviews: list[str]
    ratings: list[int] | None = None
    product_id: str | None = None


class RefundCheckRequest(BaseModel):
    product_name: str = Field(..., min_length=2)
    damage_image: str | None = None
    product_id: str | None = None
    order_id: str | None = None
    refund_amount: float | None = Field(default=None, ge=0)
    claim_reason: str | None = None


# ============================================================================
# Helper Functions
# ============================================================================


def build_review_payload(prediction_result, request_data, product_id, reviewer_id, review_text, rating):
    """Build a payload for storing a review prediction in the database."""
    return {
        "product_id": product_id,
        "reviewer_id": reviewer_id,
        "review_text": review_text,
        "rating": rating,
        "prediction": prediction_result["prediction"],
        "fake_probability": prediction_result["fake_probability"],
        "text_pattern_score": prediction_result["text_pattern_score"],
        "sentiment_mismatch": 0.0,
        "similarity_score": prediction_result["similarity_score"],
        "trust_score": prediction_result["trust_score"],
        "explanation": prediction_result["explanation"],
        "suspicious_words": prediction_result["suspicious_words"],
    }


def save_review_to_db(db: Session, payload: dict):
    """Save a review prediction to the database."""
    from database import store_review_prediction, upsert_reviewer_profile

    try:
        record = store_review_prediction(db, payload)
        if payload.get("reviewer_id"):
            upsert_reviewer_profile(
                db,
                payload["reviewer_id"],
                payload["rating"],
                payload["fake_probability"],
            )
        return record
    except Exception as exc:
        logger.warning("Could not save review: %s", exc)
        return None


def save_refund_to_db(db: Session, payload: dict):
    """Save a refund check to the database."""
    from database import store_refund_check

    try:
        record = store_refund_check(db, payload)
        return record
    except Exception as exc:
        logger.warning("Could not save refund check: %s", exc)
        return None


# ============================================================================
# FastAPI Application and Routes
# ============================================================================

app = FastAPI(
    title="Review Checker Backend",
    version=APP_VERSION,
    description="Machine-learning fake review detection with FastAPI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """Check if the backend is ready."""
    dataset_path = str(PROJECT_DIR / "dataset" / "sentiment.csv")
    return HealthCheckResponse(
        status="healthy",
        version=APP_VERSION,
        dataset_path=dataset_path,
        model_ready=True,
        database_ready=db_ready,
    )


@app.post("/api/analyze-review")
def analyze_review(request: ReviewAnalysisRequest, db: Session = Depends(get_db)):
    """Analyze a single review and return a prediction."""
    result = predict_review(request.review_text, request.rating)

    payload = build_review_payload(
        result,
        request,
        request.product_id,
        request.reviewer_id,
        request.review_text,
        request.rating,
    )

    record = save_review_to_db(db, payload)
    if record:
        result["record_id"] = record.id
        result["database_saved"] = True
    else:
        result["record_id"] = None
        result["database_saved"] = False

    return result


@app.post("/api/extract-product-reviews")
def extract_product_reviews(request: ProductReviewExtractionRequest):
    """Extract reviews from a product URL."""
    try:
        return extract_reviews_from_url(request.product_url, request.max_reviews)
    except Exception as exc:
        logger.warning("Could not extract reviews: %s", exc)
        raise HTTPException(status_code=400, detail="Could not extract reviews from this link.")


@app.post("/api/analyze-product-reviews")
def analyze_product_reviews(request: AnalyzeProductReviewsRequest, db: Session = Depends(get_db)):
    """Analyze multiple reviews for a product."""
    result = summarize_product_reviews(
        request.reviews,
        ratings=request.ratings,
        product_url=request.product_url,
        product_id=request.product_id,
    )

    # Save each review to the database
    for item in result.get("review_results", []):
        payload = build_review_payload(
            item,
            request,
            request.product_id,
            None,
            item["review_text"],
            item["rating"],
        )
        save_review_to_db(db, payload)

    return result


@app.post("/api/refund-check")
def refund_check(request: RefundCheckRequest, db: Session = Depends(get_db)):
    """Check a refund claim for fraud risk."""
    from database import get_refund_hashes

    existing_hashes = []
    try:
        existing_hashes = get_refund_hashes(db)
    except Exception as exc:
        logger.warning("Could not load refund hashes: %s", exc)

    result = analyze_refund_fraud(
        product_name=request.product_name,
        damage_image=request.damage_image,
        refund_amount=request.refund_amount,
        existing_hashes=existing_hashes,
    )

    payload = {
        "product_name": request.product_name,
        "product_id": request.product_id,
        "order_id": request.order_id,
        "claim_reason": request.claim_reason,
        "refund_amount": request.refund_amount,
        "image_hash": result["image_hash"],
        "image_pattern_score": result["image_pattern_score"],
        "image_similarity_score": result["image_similarity_score"],
        "product_match_result": result["product_match_result"],
        "fraud_score": result["fraud_score"],
        "risk_level": result["risk_level"],
    }

    record = save_refund_to_db(db, payload)
    if record:
        result["record_id"] = record.id
        result["database_saved"] = True
    else:
        result["record_id"] = None
        result["database_saved"] = False

    return result


@app.get("/api/analytics")
def analytics(time_range: str = "week", db: Session = Depends(get_db)):
    """Get analytics summary for a time range."""
    from database import get_analytics_summary

    days = {"day": 1, "week": 7, "month": 30, "year": 365}.get(time_range, 7)
    try:
        return get_analytics_summary(db, days=days)
    except Exception as exc:
        logger.warning("Could not load analytics: %s", exc)
        return {
            "overview": {
                "totalReviews": 0,
                "fakeReviewsDetected": 0,
                "refundFraudChecks": 0,
                "highRiskRefunds": 0,
                "averageTrustScore": 0,
                "fakeReviewPercentage": 0,
                "suspiciousReviewerScore": 0,
                "productTrustScore": 0,
            },
            "barChart": [],
            "pieChart": [],
            "lineChart": [],
            "topReviewers": [],
        }


@app.get("/api/product/{product_id}/trust-summary")
def product_trust_summary(product_id: str, db: Session = Depends(get_db)):
    """Get trust summary for a specific product."""
    from database import get_product_trust_summary

    try:
        return get_product_trust_summary(db, product_id)
    except Exception as exc:
        logger.warning("Could not load product trust summary: %s", exc)
        return {
            "product_id": product_id,
            "product_trust_score": 0.0,
            "fake_review_percentage": 0.0,
            "buy_recommendation": "insufficient data",
            "summary": "No saved reviews were found for this product.",
            "review_volume": 0,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
