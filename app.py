import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

from dotenv import load_dotenv

load_dotenv()


import os

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def fetch_movie_poster(movie_title):
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": movie_title, "language": "en-US"}

        response = requests.get(search_url, params=params, timeout=15)

        if response.status_code != 200:
            st.error(f"API error: {response.status_code}")
            return None

        data = response.json()
        results = data.get("results", [])

        if not results:
            st.warning(f"No results found for: {movie_title}")
            return None

        movie = results[0]
        poster_path = movie.get("poster_path")

        if poster_path:
            return TMDB_IMAGE_BASE_URL + poster_path

        st.warning("No poster available for this movie")
        return None

    except Exception as e:
        st.error(f"Error fetching poster: {str(e)}")
        return None


st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("Movie Recommendation System")
st.write("Get movie recommendations based on similar content.")


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/tmdb_5000_movies.csv")
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["combined_features"] = df["genres"] + " " + df["overview"]
    return df


df = load_data()


# TF-IDF Vectorization
@st.cache_data
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = compute_similarity(df)

# Movie index mapping
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()


def get_movie_overview(movie_title):
    overview = df[df["title"] == movie_title]["overview"].values
    if len(overview) > 0:
        return overview[0]
    return "No overview available."


num_recs = st.slider(
    "Number of recommendations", min_value=5, max_value=15, value=5, step=1
)


# Recommendation function
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in indices:
        return []

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    return df["title"].iloc[movie_indices]


# User input
movie_name = st.selectbox("Select a movie", sorted(df["title"].unique()))

st.subheader("Movie Overview")
st.write(get_movie_overview(movie_name))

poster_url = fetch_movie_poster(movie_name)

if poster_url:
    st.image(poster_url, width=300)
else:
    st.info("Poster not available")

recommendations = recommend_movies(movie_name, num_recs)

st.subheader("Recommended Movies:")
for movie in recommendations:
    st.write("--", movie)
