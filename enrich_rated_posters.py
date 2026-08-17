import pandas as pd
import requests
import os
import time
import sys
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv(dotenv_path="key.env")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    print("[ERROR] TMDB_API_KEY not found in key.env!")
    sys.exit(1)

print(f"Loaded TMDB_API_KEY: {TMDB_API_KEY[:4]}...{TMDB_API_KEY[-4:]}")

csv_path = "movies_clean.csv"
df = pd.read_csv(csv_path)

# Filter for rated movies with missing/empty poster_url
mask_missing = df["poster_url"].isna() | (df["poster_url"].astype(str).str.strip() == "")
mask_rated = df["rating"].notna()
missing_rated = df[mask_rated & mask_missing].copy()

total_missing = len(missing_rated)
print(f"Found {total_missing} rated movies missing poster URLs.")

def clean_title(t):
    return "".join(c.lower() for c in str(t) if c.isalnum())

def search_poster(title, year=None):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        results = data.get("results")
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
        clean_movie_title = clean_title(movie_title)
        
        # Check if the title is similar
        title_match = (clean_target in clean_movie_title) or (clean_movie_title in clean_target)
        if not title_match:
            continue
            
        release_date = movie.get("release_date")
        if not release_date:
            if clean_target == clean_movie_title and best_match is None:
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

def fetch_worker(item):
    idx, title, year = item
    url = search_poster(title, year)
    return idx, title, url

tasks = []
for idx, row in missing_rated.iterrows():
    tasks.append((idx, row["title"], row["year"]))

success_count = 0
start_time = time.time()

print("Starting concurrent poster pre-fetching (20 threads)...")
sys.stdout.flush()

# Use ThreadPoolExecutor with 20 threads
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(fetch_worker, task): task for task in tasks}
    
    for count, future in enumerate(as_completed(futures), 1):
        idx, title, poster_url = future.result()
        if poster_url:
            df.at[idx, "poster_url"] = poster_url
            success_count += 1
            print(f"[{success_count}] Match: '{title}' -> {poster_url}")
            sys.stdout.flush()
        else:
            # Print a dot for progress of misses
            print(".", end="")
            sys.stdout.flush()
            if count % 50 == 0:
                print(f" {count}/{total_missing} processed")
                sys.stdout.flush()

        # Save progress every 50 processed items
        if count % 50 == 0:
            df.to_csv(csv_path, index=False)

# Final save
df.to_csv(csv_path, index=False)
end_time = time.time()

print("\n" + "="*40)
print(f"Pre-fetching complete!")
print(f"Successfully enriched: {success_count} / {total_missing} movies.")
print(f"Time taken: {end_time - start_time:.2f} seconds.")
print("="*40)
