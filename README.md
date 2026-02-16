# Hybrid Recommendation System

A hybrid recommendation engine combining **Collaborative Filtering** and **Content-Based Filtering** with vectorization analysis for recommending opportunities (jobs, internships, courses, grants, events, freelance gigs) to users.

## 📋 Table of Contents

- [What the Project Does](#what-the-project-does)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)

---

## What the Project Does

### Problem Statement

Traditional recommendation systems often struggle with:
- **Cold-start problem**: New users or items with no interaction history
- **Limited personalization**: Single-method approaches miss nuanced user preferences
- **Lack of explainability**: Users don't understand why recommendations are made

### Solution Overview

This system addresses these challenges by:

1. **Hybrid Recommendation Strategy**
   - Combines collaborative filtering (user behavior patterns) with content-based filtering (item attributes)
   - Automatically switches strategies based on data availability (cold-start handling)

2. **Content-Based Filtering (CBF)**
   - Uses TF-IDF vectorization for text similarity
   - Analyzes opportunity descriptions, skills, categories
   - Cosine similarity for matching user profiles to opportunities

3. **Collaborative Filtering (CF)**
   - Item-based implicit feedback approach
   - Learns from user interaction patterns (views, clicks, saves, applications)
   - No explicit ratings required

4. **Vectorization Analysis**
   - Compares TF-IDF, TF-IDF+SVD, and SBERT embeddings
   - Generates performance reports for technique selection

### Recommendation Approach

The system uses a **weighted hybrid approach**:

- **Content-based signals**: Text similarity between user profile/query and opportunity descriptions
- **Collaborative signals**: Similar users' interaction patterns
- **Hybrid combination**: Weighted fusion (configurable `alpha` parameter)
- **Cold-start handling**: Falls back to content-based or popularity when collaborative data is insufficient

---

## How to Run

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation Steps

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

#### 1. Start the FastAPI Backend

```bash
# From the project root directory
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`

- API docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Alternative docs (ReDoc): `http://127.0.0.1:8000/redoc`

#### 2. Start the Streamlit Dashboard

In a **new terminal** (keep the API running):

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

#### 3. Run Evaluation Scripts

**Offline Evaluation** (Precision@K, Recall@K, NDCG, MRR):
```bash
python -m src.evaluation.evaluate
```

**Vectorization Comparison** (generates HTML report):
```bash
python -m src.evaluation.vectorization_compare
```

The report will be saved to `reports/vectorization_report.html`

### Running with Docker

*(Docker setup not included in MVP - see Future Improvements)*

### Accessing Deployed App

1. **API Endpoints**:
   - `GET /` - Health check
   - `POST /recommend` - Get recommendations
   - `POST /users` - Create/update user
   - `POST /opportunities` - Add opportunity
   - `POST /interactions` - Record interaction
   - `POST /reload` - Reload models (after data changes)

2. **Dashboard Pages**:
   - **Recommendations**: Get personalized recommendations
   - **Add Opportunity**: Ingest new opportunities
   - **Add User**: Create user profiles
   - **Record Interaction**: Track user behavior

### Example API Usage

**Get Recommendations**:
```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python developer",
    "user_id": 1,
    "k": 5,
    "alpha": 0.6
  }'
```

**Add Opportunity**:
```bash
curl -X POST "http://127.0.0.1:8000/opportunities" \
  -H "Content-Type: application/json" \
  -d '{
    "opportunity_id": 100,
    "title": "Senior ML Engineer",
    "description": "Build ML models for production",
    "skills_required": ["Python", "TensorFlow", "MLOps"],
    "category": "Technology",
    "location": "Remote",
    "opportunity_type": "job"
  }'
```

---

## Limitations

### Dataset Size Constraints

- **Small datasets**: Performance may be limited with <100 users or <50 opportunities
- **Sparse interactions**: Requires sufficient interaction density for collaborative filtering to be effective
- **Memory**: Large datasets (>100K items) may require optimization (e.g., approximate nearest neighbors)

### Cold-Start Issues

- **New users**: Recommendations rely on content-based matching until sufficient interaction history
- **New items**: May not appear in collaborative recommendations until users interact with them
- **Mitigation**: System automatically falls back to content-based or popularity-based recommendations

### Offline Evaluation Limitations

- **Train/test split**: Uses simple random split (no temporal validation)
- **Implicit feedback**: Assumes all interactions are positive (no negative signals)
- **No A/B testing**: Evaluation is offline only (no real-world user feedback)

### Other Limitations

- **Model refresh**: Requires API restart or `/reload` endpoint call after data changes
- **No real-time updates**: Recommendations don't update instantly as new interactions occur
- **Single database**: SQLite (not suitable for high-concurrency production)

---

## Future Improvements

### Online Learning
- Incremental model updates as new interactions arrive
- Real-time recommendation refresh without full retraining

### Real-Time Personalization
- Streaming recommendation updates
- Session-based recommendations

### Advanced Embeddings
- Fine-tuned domain-specific embeddings
- Multi-modal features (images, structured data)

### A/B Testing
- Framework for comparing recommendation strategies
- User feedback collection and analysis

### Larger-Scale Deployment
- PostgreSQL/MongoDB for production databases
- Redis caching for recommendation results
- Horizontal scaling with load balancers
- Kubernetes deployment configurations

### Additional Features
- **Explainability**: Detailed feature contribution analysis
- **Feedback loop**: User ratings and explicit feedback
- **Monitoring**: Model drift detection, performance tracking
- **Security**: Authentication, rate limiting, input validation

---

## Project Structure

```
Opportunity_Recommendation_System/
├── api/                    # FastAPI application
│   ├── app.py             # Main API endpoints
│   ├── deps.py            # Dependency injection (model loading)
│   └── schemas.py         # Pydantic models
├── src/
│   ├── core/              # Core recommendation algorithms
│   │   ├── content.py    # Content-based recommender
│   │   ├── collaborative.py  # Item-item CF
│   │   ├── hybrid.py      # Hybrid combination logic
│   │   ├── similarity.py # Similarity functions
│   │   └── vectorizer.py # TF-IDF, SBERT vectorizers
│   ├── data/              # Data layer
│   │   ├── db.py         # SQLite operations
│   │   └── matrices.py   # Interaction matrix building
│   ├── evaluation/        # Evaluation scripts
│   │   ├── evaluate.py   # Offline evaluation
│   │   ├── metrics.py    # Precision, Recall, NDCG, F1, RMSE, MAE
│   │   └── vectorization_compare.py  # Vectorization comparison
│   ├── pipelines/         # High-level pipelines
│   │   └── recommender.py  # Hybrid recommender pipeline
│   └── main.py            # Utility functions
├── data/                  # Data files
│   ├── app.db            # SQLite database (created on first run)
│   ├── opportunities.csv  # Sample opportunities (seeded)
│   └── interactions.csv  # Sample interactions (seeded)
├── reports/              # Generated reports
│   └── vectorization_report.html
├── dashboard.py          # Streamlit dashboard
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## API Documentation

### POST /recommend

Get personalized recommendations.

**Request Body**:
```json
{
  "query": "Python developer",  // Optional: text query
  "user_id": 1,                 // Optional: user ID
  "k": 5,                       // Number of recommendations
  "alpha": 0.5,                 // Hybrid weight (0=CF, 1=CBF)
  "include_reasons": true        // Include explanation
}
```

**Response**:
```json
{
  "results": [
    {
      "opportunity_id": 1,
      "title": "ML Engineer",
      "description": "...",
      "score": 0.85,
      "reason": "Content match (text similarity)"
    }
  ]
}
```

### POST /users

Create or update a user profile.

**Request Body**:
```json
{
  "user_id": 1,
  "skills": ["Python", "ML"],
  "interests": ["AI", "Data Science"],
  "experience": "5 years in ML",
  "profile_text": "ML engineer looking for..."
}
```

### POST /opportunities

Add a new opportunity.

**Request Body**:
```json
{
  "opportunity_id": 100,
  "title": "Senior ML Engineer",
  "description": "...",
  "skills_required": ["Python", "TensorFlow"],
  "category": "Technology",
  "location": "Remote",
  "opportunity_type": "job"
}
```

**Note**: After adding opportunities, call `POST /reload` to refresh models.

### POST /interactions

Record a user interaction.

**Request Body**:
```json
{
  "user_id": 1,
  "opportunity_id": 100,
  "event_type": "view",  // view, click, save, apply, implicit
  "weight": 1.0
}
```

### POST /reload

Reload recommendation models (useful after data changes).

**Response**:
```json
{
  "status": "OK",
  "note": "Models will be rebuilt on next request to /recommend."
}
```

---