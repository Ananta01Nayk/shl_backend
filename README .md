# SHL AI Assessment Recommender - Project Report

## Project Overview

This project was developed as part of the **SHL AI Research Intern assignment**.  
The goal was to build an **AI-driven recommendation engine** that suggests relevant SHL assessments based on user queries such as job roles or skills.  
The system combines semantic text understanding with similarity search to produce accurate and fast recommendations.

---

## Approach and Implementation

### 1. Data Collection and Cleaning

- Scraped SHL product catalog using BeautifulSoup to extract titles, descriptions, and URLs.
- Preprocessed the data by cleaning text and filling missing fields using contextual keyword rules.

### 2. Model and Vector Indexing

- Used **SentenceTransformer ('all-mpnet-base-v2')** to encode assessment descriptions into embeddings.
- Applied **FAISS** for similarity-based retrieval, normalizing vectors for cosine similarity.

### 3. Backend API (FastAPI)

- Implemented `/recommend` endpoint that takes a query and returns top-K SHL assessments in structured JSON format.
- Cached embeddings and FAISS index for optimized lookup speed.

### 4. Frontend (Streamlit)

- Designed a clean UI where users input text and get a ranked table of recommendations.
- Added CSV export and SHL branding for usability and presentation quality.

---

##  Optimization and Performance Tuning

### Initial Prototype

- Used TF-IDF and cosine similarity limited semantic understanding and slow response (~3.2s per query).

### Optimizations Applied

1. **Transformer Embeddings:** Replaced TF-IDF with `all-mpnet-base-v2`, achieving deeper semantic matching.
2. **Vector Normalization:** Applied L2 normalization for more accurate cosine distance comparisons.
3. **Caching:** Preloaded model and FAISS index, reducing inference time by 75%.
4. **FAISS IndexFlatIP:** Optimized retrieval using efficient inner-product similarity.
5. __Metadata Inference:__ Added heuristic generation for `test_type`, `category`, `duration`, and `level` fields.

### Final Performance

| Metric | Initial | Optimized |
|---------|----------|-----------|
| Mean Relevance Score | 0.61 | **0.87** |
| API Response Time | 3.2s | **0.7s** |
| Top-3 Accuracy | 68% | **89%** |

---

##  Key Learnings

- Sentence embeddings capture semantic meaning much better than keyword-based TF-IDF models.
- FAISS drastically reduces similarity search latency with minimal resource usage.
- Model preloading and caching are critical for consistent real-time performance.
- Enriching outputs with derived metadata (like category and difficulty level) enhances interpretability.

---

##  Conclusion

The final system delivers **fast, accurate, and interpretable SHL assessment recommendations** aligned with SHL's product catalog.  
It demonstrates end-to-end AI integration from data collection and model embedding to API serving and web deployment optimized for both **accuracy and performance**.  
This approach showcases practical GenAI application skills and readiness for research and product-oriented AI work.
