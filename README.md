
# Nollywood Movie Recommender

A content-based and genre recommendation system for Nollywood films, built end-to-end with a Python/pandas/scikit-learn data pipeline, a FastAPI backend, and a modern glassmorphism web frontend — complete with movie posters fetched dynamically from TMDB.

Built by **ALIYU**.

---

## Overview

This project lets users explore over 7,300 Nigerian films with:
1. **Genre Search**: Instant search by genre (e.g. "Comedy", "Drama", "Romance", "Action").
2. **AI Content Similarity ("More Like This")**: Recommends movies similar to any selected movie using TF-IDF vectorization and cosine similarity across genres, directors, cast members, and plot overviews.
3. **Multi-Field Search**: Flexible searching across movie Title, Genre, Director, and Cast/Actors simultaneously.
4. **Interactive Glassmorphism UI**: Dynamic movie detail modals, quick filter pills, responsive card grid, and local Watchlist/Bookmark saving.

---

## Tech Stack

| Layer | Technologies & Tools |
|---|---|
| **Data Processing & ML** | Python, pandas, NumPy, scikit-learn (`TfidfVectorizer`, Cosine Similarity) |
| **Backend / API** | FastAPI, Uvicorn, python-dotenv |
| **Poster Enrichment & APIs** | TMDB (The Movie Database) API, `requests`, `concurrent.futures` (`ThreadPoolExecutor`) |
| **Frontend** | HTML5, Vanilla CSS3 (Glassmorphism, CSS Grid & Flexbox), Vanilla JavaScript (ES6+, Fetch API, LocalStorage), FontAwesome 6, Google Fonts (Plus Jakarta Sans) |
| **Data Source** | Nollywood Movies Dataset (scraped from IMDb) |
| **Deployment / Serverless** | Vercel (`vercel.json`), Python ASGI (`api/index.py`) |

---

## Project Structure

```
nollywood-recommender/
├── api/
│   └── index.py              # Vercel serverless entry point
├── enrich_rated_posters.py   # Multi-threaded TMDB poster pre-fetch script
├── evaluation.py             # Recommender evaluation and test suite
├── index.html                # Modern glassmorphism web frontend with Watchlist & Modals
├── key.env                   # TMDB API key configuration
├── main.py                   # FastAPI backend with TF-IDF similarity & search endpoints
├── movierecommender.ipynb    # Data cleaning, EDA, and poster-fetching notebook
├── movies_clean.csv          # Cleaned Nollywood movie dataset with poster URLs
├── requirements.txt          # Python dependencies (fastapi, pandas, scikit-learn, etc.)
├── vercel.json               # Vercel deployment configuration
└── README.md                 # Project documentation
```

---

## API Reference

### 1. `GET /recommend`
Recommends movies by genre.
- **Parameters**: `genre` (string, required), `limit` (int, default 10)
- **Example**: `/recommend?genre=Comedy&limit=10`

### 2. `GET /recommend/similar`
Content-based recommendation engine powered by TF-IDF vectorization & cosine similarity.
- **Parameters**: `title` (string, required), `limit` (int, default 10)
- **Example**: `/recommend/similar?title=Daughters%20of%20Donald&limit=10`

### 3. `GET /search`
Multi-field search across title, genre, director, stars, or description.
- **Parameters**: `query` (string, required), `field` (string: `all`, `title`, `genre`, `director`, `stars`), `limit` (int)
- **Example**: `/search?query=Genevieve&field=stars&limit=10`

### 4. `GET /featured`
Returns top-rated featured Nollywood movies.

---

## Running Locally

### Requirements
- Python 3.10+
- `pip install -r requirements.txt`

### Start Backend Server
```bash
uvicorn main:app --reload
```
Access interactive API documentation at: `http://127.0.0.1:8000/docs`

### Open Web Frontend
Open `index.html` in your browser.

---

Built with FastAPI + pandas + scikit-learn — Nollywood data sourced from IMDb, posters sourced from TMDB.
=======
# Nollywood-movie-recommender-3mtt-
A nollywood movie recommender 
>>>>>>> 50868245665d856c4d669a6caed630ca44efea9a
