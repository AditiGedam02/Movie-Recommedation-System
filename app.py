import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("Movie Recommendation System")
st.write("Get movie recommendations based on similar content.")


@st.cache_data
def load_data():
    df = pd.read_csv("data/tmdb_5000_movies.csv")
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["combined_features"] = df["genres"] + " " + df["overview"]
    return df


df = load_data()


@st.cache_data
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = compute_similarity(df)

indices = pd.Series(df.index, index=df["title"]).drop_duplicates()


def get_movie_overview(movie_title):
    overview = df[df["title"] == movie_title]["overview"].values
    if len(overview) > 0:
        return overview[0]
    return "No overview available."


num_recs = st.slider(
    "Number of recommendations", min_value=5, max_value=15, value=5, step=1
)


def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in indices:
        return []

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    return df["title"].iloc[movie_indices]


movie_name = st.selectbox("Select a movie", sorted(df["title"].unique()))

st.subheader("Movie Overview")
st.write(get_movie_overview(movie_name))

recommendations = recommend_movies(movie_name, num_recs)

st.subheader("Recommended Movies:")
for movie in recommendations:
    st.write("--", movie)
