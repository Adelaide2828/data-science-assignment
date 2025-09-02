import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Load dataset
df = pd.read_csv("movies11.csv")
df.columns = df.columns.str.strip()
print("Dataset head:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# Check if we have genre data
if 'genre' not in df.columns:
    print("Error: 'genre' column not found. Please check your dataset.")
else:
    # Create a movie-genre matrix (content features)
    print("\nSample genres:", df['genre'].iloc[:5].tolist())
    
    # Clean the genre data
    df['genre'] = df['genre'].fillna('').astype(str)
    
    # Approach: Use TF-IDF for genre processing (works with various formats)
    tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
    genre_matrix = tfidf.fit_transform(df['genre'])
    genre_features = pd.DataFrame(genre_matrix.toarray(), 
                                 columns=tfidf.get_feature_names_out(), 
                                 index=df['movie_title'])
    
    print("\nGenre features matrix shape:", genre_features.shape)
    print("Genre features columns:", genre_features.columns.tolist()[:10])  # Show first 10 features
    
    # Compute movie similarity based on genres
    movie_similarity = cosine_similarity(genre_matrix)
    movie_similarity_df = pd.DataFrame(movie_similarity, 
                                      index=df['movie_title'], 
                                      columns=df['movie_title'])
    
    print("\nMovie similarity matrix shape:", movie_similarity_df.shape)
    
    # Function to get content-based recommendations (FIXED)
    def get_content_recommendations(movie_title, n_recommendations=5):
        if movie_title not in movie_similarity_df.columns:
            return f"Movie '{movie_title}' not found in dataset"
        
        # Get similarity scores for the movie - this is now a Series
        similar_scores = movie_similarity_df[movie_title]
        
        # Sort by similarity (descending) and exclude the movie itself
        # For Series, we don't need the 'by' parameter
        similar_movies = similar_scores.sort_values(ascending=False)
        
        # Exclude the movie itself and get top N recommendations
        similar_movies = similar_movies[similar_movies.index != movie_title]
        top_recommendations = similar_movies.head(n_recommendations)
        
        return top_recommendations.index.tolist()
    
    # Example: Get recommendations based on a movie
    test_movie = df['movie_title'].iloc[0]  # Use the first movie as example
    print(f"\nMovies similar to '{test_movie}':")
    recommendations = get_content_recommendations(test_movie, 3)
    
    if isinstance(recommendations, list):
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")
    else:
        print(recommendations)
    
    # Additional: Get recommendations for a specific user
    def get_user_content_recommendations(user_id, n_recommendations=5):
        # Get movies the user has rated highly (e.g., rating >= 4)
        user_movies = df[(df['user_id'] == user_id) & (df['rating'] >= 4)]['movie_title'].tolist()
        
        if not user_movies:
            return f"User {user_id} has no highly rated movies for recommendations"
        
        # Get recommendations based on all user's liked movies
        all_recommendations = []
        for movie in user_movies:
            if movie in movie_similarity_df.columns:
                recs = get_content_recommendations(movie, n_recommendations * 2)  # Get more to filter
                all_recommendations.extend(recs)
        
        # Remove duplicates and movies user has already watched
        user_watched = df[df['user_id'] == user_id]['movie_title'].tolist()
        unique_recommendations = [movie for movie in set(all_recommendations) 
                                 if movie not in user_watched]
        
        return unique_recommendations[:n_recommendations]
    
    # Example: Get content-based recommendations for user 1
    user_id = 5
    print(f"\nContent-based recommendations for user {user_id}:")
    user_recs = get_user_content_recommendations(user_id, 4)
    
    if isinstance(user_recs, list):
        for i, movie in enumerate(user_recs, 1):
            print(f"{i}. {movie}")
    else:
        print(user_recs)

    # Bonus: Show the similarity scores for transparency
    print(f"\nSimilarity scores for '{test_movie}':")
    test_scores = movie_similarity_df[test_movie].sort_values(ascending=False)
    for movie, score in test_scores.head(6).items():  # Show top 6 including itself
        print(f"{movie}: {score:.3f}")
