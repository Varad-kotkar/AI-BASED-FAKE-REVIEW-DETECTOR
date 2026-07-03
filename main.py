"""Fake Review Checker - FastAPI backend"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import init_db, get_db, save_review, get_analytics
from ml_logic import predict_review, summarize_product_reviews

app = FastAPI(title="Fake Review Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# create the database table when the app starts
init_db()


# ---------------------------
# Request models
# ---------------------------

class ReviewRequest(BaseModel):
    review_text: str
    rating: int
    product_id: str = "product-demo"


class MultipleReviewsRequest(BaseModel):
    reviews: list[str]
    ratings: list[int] = []
    product_id: str = "product-demo"


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/analyze-review")
def analyze_review(request: ReviewRequest):
    result = predict_review(request.review_text, request.rating)

    db = get_db()
    save_review(db, request.product_id, result)
    db.close()

    return result


@app.post("/api/analyze-product-reviews")
def analyze_product_reviews(request: MultipleReviewsRequest):
    result = summarize_product_reviews(request.reviews, request.ratings)

    db = get_db()
    for review_result in result["review_results"]:
        save_review(db, request.product_id, review_result)
    db.close()

    return result


@app.get("/api/analytics")
def analytics():
    db = get_db()
    data = get_analytics(db)
    db.close()
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
