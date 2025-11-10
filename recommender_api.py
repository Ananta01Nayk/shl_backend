# recommender_api.py

# SHL Assessment Recommendation API
# FastAPI backend that returns top-K SHL assessments
# and also saves each response as a JSON file

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import faiss, pickle, numpy as np, os
from sentence_transformers import SentenceTransformer
import uvicorn

# Configuration
index_data_file = "faiss_products.index"
meta_data_file = "product_records.pkl"
model_name = "all-mpnet-base-v2"

app = FastAPI(title="Ananta SHL Assessment Recommender")

# Pydantic Models

class InputQuery(BaseModel):
    query_text: str
    top_k: int = 5


class RecommendationItem(BaseModel):
    assessment_name: str
    assessment_url: str
    score: float
    test_type: str = ""
    category: str = ""
    duration: str = ""
    level: str = ""

# Globals

vector_index = None
product_records = []
encoder = None

# Load FAISS and model

def load_resources():
    global vector_index, product_records, encoder
    if vector_index is None:
        if not os.path.exists(index_data_file) or not os.path.exists(meta_data_file):
            raise FileNotFoundError("Index or metadata missing. Run create_vector_index.py first.")
        vector_index = faiss.read_index(index_data_file)
        with open(meta_data_file, "rb") as f:
            product_records = pickle.load(f)
        encoder = SentenceTransformer(model_name)
        print("✅ Resources loaded: Encoder + FAISS index + Metadata.")


@app.on_event("startup")
def startup_event():
    load_resources()

# Routes
@app.get("/")
def home():
    return {"message": "Welcome to Ananta's SHL Assessment Recommender API"}


@app.get("/health")
def health_check():
    return {"status": "ok", "items_indexed": len(product_records)}

@app.post("/recommend")
def get_recommendations(payload: InputQuery) -> Dict:
    """
    Generate top-K SHL assessment recommendations for a given query text.
    Returns SHL-compliant JSON with additional metadata fields.
    """
    if not payload.query_text or payload.top_k < 1:
        raise HTTPException(status_code=400, detail="Invalid input")

    # Encode query text
    query_vec = encoder.encode([payload.query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    # Search in FAISS index
    n_to_fetch = min(50, len(product_records))
    D, I = vector_index.search(query_vec, n_to_fetch)

    recommendations = []

    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        rec = product_records[idx]

        title = str(rec.get("title", "") or "")
        url = str(rec.get("url", "") or "")
        desc = str(rec.get("description", "") or "")
        test_type = str(rec.get("test_type", "") or "")

        # Derived fields: infer missing fields from text

        text = (title + " " + desc).lower()

        # Category inference
        if any(word in text for word in ["coding", "technical", "developer", "engineer", "programming"]):
            category = "Technical Skills"
        elif any(word in text for word in ["leadership", "manager", "management"]):
            category = "Leadership"
        elif any(word in text for word in ["communication", "teamwork", "interpersonal"]):
            category = "Soft Skills"
        elif any(word in text for word in ["cognitive", "aptitude", "verbal", "numerical"]):
            category = "Cognitive Ability"
        else:
            category = "General Assessment"

        # Level inference
        if any(word in text for word in ["advanced", "senior", "expert"]):
            level = "Senior"
        elif any(word in text for word in ["intermediate", "mid"]):
            level = "Mid"
        else:
            level = "Entry"

        # Duration inference
        if any(word in text for word in ["simulation", "comprehensive", "battery", "leadership"]):
            duration = "45 mins"
        elif any(word in text for word in ["quick", "short", "mini", "fast"]):
            duration = "15 mins"
        else:
            duration = "30 mins"

        recommendations.append({
            "assessment_name": title,
            "assessment_url": url,
            "score": float(score),
            "test_type": test_type,
            "category": category,
            "duration": duration,
            "level": level
        })

    # Limit to top_k
    recommendations = recommendations[: payload.top_k]

    return {
        "query": payload.query_text,
        "recommendations": recommendations
    }

# Run locally
if __name__ == "__main__":
    uvicorn.run("recommender_api:app", host="0.0.0.0", port=8000)
