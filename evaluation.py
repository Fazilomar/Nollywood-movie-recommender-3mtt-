import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def evaluate_recommender(data_path="movies_clean.csv"):
    print("=== STARTING EXTENDED RECOMMENDER EVALUATION ===\n")

    # 1. DATASET VALIDATION
    if not os.path.exists(data_path):
        print(f"[FAIL] Data file '{data_path}' not found!")
        return

    df = pd.read_csv(data_path)
    total_movies = len(df)
    print(f"[PASS] Data loaded successfully. Total movies: {total_movies}")

    # Check missing values in critical columns
    missing_genres = df["genre"].isna().sum()
    missing_ratings = df["rating"].isna().sum()
    print(f" - Missing Genres: {missing_genres} ({(missing_genres/total_movies)*100:.2f}%)")
    print(f" - Missing Ratings: {missing_ratings} ({(missing_ratings/total_movies)*100:.2f}%)")

    # 2. POSTER COVERAGE EVALUATION
    if "poster_url" in df.columns:
        valid_posters = df["poster_url"].notna().sum()
        poster_coverage = (valid_posters / total_movies) * 100
        print(f"[INFO] Poster URL Coverage: {valid_posters}/{total_movies} ({poster_coverage:.2f}%)")
    else:
        print("[WARN] 'poster_url' column is missing from the dataset!")

    print("\n" + "-" * 50 + "\n")

    # 3. FASTAPI CORE ENGINE INTEGRATION TEST
    print("Testing API Core Functions (TF-IDF & Cosine Similarity)...")
    try:
        import main
        
        # Test 3a: Genre recommendations
        rec_genre = main.recommend_by_genre("Comedy", num_recommendations=5)
        print(f"[PASS] recommend_by_genre('Comedy'): returned {len(rec_genre)} movies.")
        
        # Test 3b: Similar movie recommendations (TF-IDF)
        sample_title = rec_genre.iloc[0]["title"]
        similar_movies, target_title = main.recommend_similar_movies(sample_title, num_recommendations=5)
        print(f"[PASS] recommend_similar_movies('{target_title}'): returned {len(similar_movies)} similar movies.")
        for idx, row in similar_movies.iterrows():
            print(f"    -> {row['title']} (Score: {row['similarity_score']}) | Genre: {row['genre']}")

        # Test 3c: Multi-field Search
        search_res = main.search_movies_multi("Love", field="all", limit=5)
        print(f"[PASS] search_movies_multi('Love'): returned {len(search_res)} matching movies.")

    except Exception as e:
        print(f"[FAIL] Error during core API evaluation: {e}")

    print("\n=== EVALUATION COMPLETE ===")

if __name__ == "__main__":
    evaluate_recommender("movies_clean.csv")