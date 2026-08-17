import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Try importing sklearn with fallback safety
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Load environment variables
load_dotenv("key.env")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = FastAPI(
    title="Nollywood Movie Recommender API",
    description="Content-based & genre recommendation engine for Nollywood cinema",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and prepare dataset
DATA_PATH = "movies_clean.csv"
df = pd.read_csv(DATA_PATH)

# Clean null values for text processing
df["genre"] = df["genre"].fillna("")
df["director"] = df["director"].fillna("")
df["stars"] = df["stars"].fillna("")
df["show_desc"] = df["show_desc"].fillna("")
df["poster_url"] = df["poster_url"].fillna("")

# Pre-compute combined content soup for Content-Based Filtering
def create_content_soup(row):
    genre_str = (row["genre"] + " ") * 3
    director_str = (row["director"] + " ") * 2
    stars_str = (row["stars"] + " ") * 2
    desc_str = str(row["show_desc"])
    
    if "it looks like we don't have a synopsis" in desc_str.lower():
        desc_str = ""
        
    return f"{genre_str}{director_str}{stars_str}{desc_str}".strip().lower()

df["content_soup"] = df.apply(create_content_soup, axis=1)

# Fit TF-IDF Vectorizer if sklearn is installed
tfidf_matrix = None
if SKLEARN_AVAILABLE:
    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(df["content_soup"])
    except Exception as e:
        print(f"Warning: Failed to fit TF-IDF vectorizer: {e}")
        SKLEARN_AVAILABLE = False

# Quick title index mapping
def clean_title(t):
    return "".join(c.lower() for c in str(t) if c.isalnum())

df["clean_title"] = df["title"].apply(clean_title)
title_to_index = {title: idx for idx, title in enumerate(df["clean_title"])}


# Recommender Core Functions
def recommend_by_genre(genre_input: str, num_recommendations: int = 10):
    genre_input = genre_input.strip().lower()
    mask = df["genre"].str.lower().str.contains(genre_input, na=False)
    matches = df[mask].copy()

    if matches.empty:
        return None

    matches = matches.sort_values(by="rating", ascending=False, na_position="last")
    results = matches[["title", "year", "genre", "director", "stars", "rating", "show_desc", "poster_url"]]
    return results.head(num_recommendations)


def recommend_similar_movies(title_input: str, num_recommendations: int = 10):
    clean_input = clean_title(title_input)
    
    if clean_input not in title_to_index:
        matching_idx = df[df["clean_title"].str.contains(clean_input, na=False)].index
        if matching_idx.empty:
            return None
        target_idx = matching_idx[0]
    else:
        target_idx = title_to_index[clean_input]

    target_row = df.iloc[target_idx]

    if SKLEARN_AVAILABLE and tfidf_matrix is not None:
        # Compute cosine similarity with TF-IDF
        cosine_sim = linear_kernel(tfidf_matrix[target_idx], tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [item for item in sim_scores if item[0] != target_idx][:num_recommendations]
        movie_indices = [item[0] for item in sim_scores]
        scores = [round(float(item[1]), 3) for item in sim_scores]
    else:
        # Fallback similarity based on genre & director overlap
        target_genre = target_row["genre"].lower()
        target_director = target_row["director"].lower()
        
        matches = df[df.index != target_idx].copy()
        
        def calc_fallback_score(r):
            score = 0.0
            if target_genre and target_genre in r["genre"].lower():
                score += 0.6
            if target_director and target_director != "unknown" and target_director in r["director"].lower():
                score += 0.3
            return score

        matches["similarity_score"] = matches.apply(calc_fallback_score, axis=1)
        matches = matches.sort_values(by=["similarity_score", "rating"], ascending=[False, False])
        results = matches.head(num_recommendations)[["title", "year", "genre", "director", "stars", "rating", "show_desc", "poster_url", "similarity_score"]]
        return results, target_row["title"]

    results = df.iloc[movie_indices][["title", "year", "genre", "director", "stars", "rating", "show_desc", "poster_url"]].copy()
    results["similarity_score"] = scores
    return results, target_row["title"]


def search_movies_multi(query: str, field: str = "all", limit: int = 10):
    q = query.strip().lower()
    if not q:
        return df.head(limit)

    if field == "title":
        mask = df["title"].str.lower().str.contains(q, na=False)
    elif field == "genre":
        mask = df["genre"].str.lower().str.contains(q, na=False)
    elif field == "director":
        mask = df["director"].str.lower().str.contains(q, na=False)
    elif field == "stars":
        mask = df["stars"].str.lower().str.contains(q, na=False)
    else:
        mask = (
            df["title"].str.lower().str.contains(q, na=False) |
            df["genre"].str.lower().str.contains(q, na=False) |
            df["director"].str.lower().str.contains(q, na=False) |
            df["stars"].str.lower().str.contains(q, na=False) |
            df["show_desc"].str.lower().str.contains(q, na=False)
        )

    matches = df[mask].copy()
    matches = matches.sort_values(by="rating", ascending=False, na_position="last")
    results = matches[["title", "year", "genre", "director", "stars", "rating", "show_desc", "poster_url"]]
    return results.head(limit)


# TMDB Poster Fetcher
def fetch_and_cache_poster(title, year):
    if not TMDB_API_KEY:
        return None

    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}

    try:
        response = requests.get(url, params=params, timeout=4)
        if response.status_code != 200:
            return None
        results = response.json().get("results")
    except Exception:
        return None

    if not results:
        return None

    target_year = None
    if year and pd.notna(year):
        try:
            target_year = int(float(year))
        except ValueError:
            pass

    best_match = None
    best_diff = None
    clean_target = clean_title(title)

    for movie in results:
        movie_title = movie.get("title", "")
        clean_movie = clean_title(movie_title)

        if (clean_target in clean_movie) or (clean_movie in clean_target):
            release_date = movie.get("release_date")
            if not release_date:
                if clean_target == clean_movie and best_match is None:
                    best_match = movie
                continue

            try:
                movie_year = int(release_date[:4])
            except ValueError:
                continue

            if target_year:
                diff = abs(movie_year - target_year)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_match = movie
            else:
                best_match = movie
                break

    if best_match:
        if best_diff is None or best_diff <= 2:
            poster_path = best_match.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        elif clean_title(best_match.get("title", "")) == clean_target and best_diff <= 3:
            poster_path = best_match.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    if results:
        first = results[0]
        if clean_title(first.get("title", "")) == clean_target:
            poster_path = first.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return None


def fetch_posters_background(movies_to_enrich):
    if not movies_to_enrich:
        return

    from concurrent.futures import ThreadPoolExecutor

    def worker(movie):
        url = fetch_and_cache_poster(movie["title"], movie["year"])
        return movie["title"], url

    updated_any = False
    with ThreadPoolExecutor(max_workers=min(10, len(movies_to_enrich))) as executor:
        futures = [executor.submit(worker, m) for m in movies_to_enrich]
        for future in futures:
            try:
                title, fetched_url = future.result()
                if fetched_url:
                    idx = df[df["title"] == title].index
                    if not idx.empty:
                        df.loc[idx[0], "poster_url"] = fetched_url
                        updated_any = True
            except Exception as e:
                print(f"Error fetching poster in background: {e}")

    if updated_any:
        try:
            df.to_csv(DATA_PATH, index=False)
            print("Background Task: Saved newly fetched posters to movies_clean.csv")
        except Exception as e:
            print(f"Failed to save cached posters: {e}")


# API Endpoints
@app.get("/")
def home():
    return {
        "name": "Nollywood Movie Recommender API",
        "version": "2.0.0",
        "total_movies": len(df),
        "posters_cached": int((df["poster_url"] != "").sum()),
        "sklearn_tfidf_active": SKLEARN_AVAILABLE,
        "endpoints": {
            "/recommend": "Recommend by genre",
            "/recommend/similar": "Content-based recommendations for a target movie",
            "/search": "Multi-field movie search across title, genre, director, stars",
            "/featured": "Collection of top-rated & trending Nollywood movies"
        }
    }


@app.get("/recommend")
def get_recommendations(
    background_tasks: BackgroundTasks,
    genre: str = Query(..., description="Genre to search for, e.g. Comedy"),
    limit: int = Query(10, description="How many results to return")
):
    results = recommend_by_genre(genre, limit)

    if results is None:
        return {"genre": genre, "count": 0, "message": f"No movies found for genre '{genre}'.", "results": []}

    results_clean = results.fillna("").to_dict(orient="records")

    missing_posters = [
        {"title": m["title"], "year": m["year"]}
        for m in results_clean
        if not m.get("poster_url") or str(m["poster_url"]).strip() == ""
    ]

    if missing_posters:
        background_tasks.add_task(fetch_posters_background, missing_posters)

    return {
        "genre": genre,
        "count": len(results_clean),
        "results": results_clean
    }


@app.get("/recommend/similar")
def get_similar_recommendations(
    background_tasks: BackgroundTasks,
    title: str = Query(..., description="Movie title to find recommendations for"),
    limit: int = Query(10, description="Number of recommendations")
):
    output = recommend_similar_movies(title, limit)
    if output is None:
        raise HTTPException(status_code=404, detail=f"Movie matching '{title}' was not found in dataset.")

    results, matched_title = output
    results_clean = results.fillna("").to_dict(orient="records")

    missing_posters = [
        {"title": m["title"], "year": m["year"]}
        for m in results_clean
        if not m.get("poster_url") or str(m["poster_url"]).strip() == ""
    ]

    if missing_posters:
        background_tasks.add_task(fetch_posters_background, missing_posters)

    return {
        "target_movie": matched_title,
        "count": len(results_clean),
        "results": results_clean
    }


@app.get("/search")
def search_movies(
    background_tasks: BackgroundTasks,
    query: str = Query(..., description="Search term"),
    field: str = Query("all", description="Search field: all, title, genre, director, stars"),
    limit: int = Query(10, description="Number of results")
):
    results = search_movies_multi(query, field, limit)

    if results is None or results.empty:
        return {"query": query, "field": field, "count": 0, "results": []}

    results_clean = results.fillna("").to_dict(orient="records")

    missing_posters = [
        {"title": m["title"], "year": m["year"]}
        for m in results_clean
        if not m.get("poster_url") or str(m["poster_url"]).strip() == ""
    ]

    if missing_posters:
        background_tasks.add_task(fetch_posters_background, missing_posters)

    return {
        "query": query,
        "field": field,
        "count": len(results_clean),
        "results": results_clean
    }


@app.get("/featured")
def get_featured_movies(limit: int = Query(12, description="Number of featured movies")):
    top_rated = df[df["rating"].notna()].sort_values(by="rating", ascending=False)
    featured = top_rated.head(limit)
    results_clean = featured.fillna("").to_dict(orient="records")
    
    return {
        "count": len(results_clean),
        "results": results_clean
    }