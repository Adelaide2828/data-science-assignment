
import pandas as pd

# Load dataset
df = pd.read_csv("movies.csv")
print(df.head())

df.columns = df.columns.str.strip()

#Creating a User-Item Matrix
user_item_matrix = df.pivot_table(index="user_id", columns="movie_title", values="rating")
print(user_item_matrix)

#Apply Collaborative Filtering (Cosine Similarity)
from sklearn.metrics.pairwise import cosine_similarity

# Fill NaN with 0 for similarity calculation
matrix_filled = user_item_matrix.fillna(0)

# Compute similarity between users
similarity = cosine_similarity(matrix_filled)
print("User Similarity Matrix:\n", similarity)

# Example: Recommend for user 1 based on most similar user
import numpy as np

user_index = 1  # user_id = 1
similar_users = similarity[user_index]

# Find the most similar user (excluding self)
most_similar_user = np.argsort(similar_users)[-2]

# Get movies rated by most similar user
recommended_movies = user_item_matrix.iloc[most_similar_user].dropna().index.tolist()
print(f"Recommended movies for User 1: {recommended_movies}")



