import difflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pickle



df = pd.read_csv('movies.csv')

# print(df.head())

# print(df.shape) (4803,24)

# selecting the relevant features for recommendation

selected_features = ['genres', 'keywords', 'overview', 'tagline', 'cast', 'director']

# handling missing values for selected features :
for feature in selected_features:
    df[feature] = df[feature].fillna('')

# combining all the 5 selected features
combined_features = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview'] + ' ' + df['tagline'] + ' ' + df['cast'] \
                    + ' ' + df['director']

# converting the textual data into feature vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

# Getting similarity score using cosine similarity
similarity = cosine_similarity(feature_vector)

# Getting movie name from user
movie_name = st.text_input("Enter your favourite movie name : ")

# Creating a list with all the movie names given in a dataset
list_movies = df['title'].tolist()

# Finding the close match for the movie name entered by the user
find_close_match = difflib.get_close_matches(movie_name, list_movies)

close_match = find_close_match[0]
# print(close_match)

# finding index of the movie with title
index_of_movie = df[df.title == close_match]['index'].values[0]
# print(index_of_movie)

# Getting the list of similar movies
similarity_score = list(enumerate(similarity[index_of_movie]))  # similar movies will have the higher similarity score.

# sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# printing the name of similar movies based on index when submit button is pressed by user.
print('Movies Suggested For You Are : \n')
i = 1
for movie in sorted_similar_movies:
 index = movie[0]
 title_from_index = df[df.index == index]['title'].values[0]
 if(i<30):
    print(i, '.', title_from_index)
    i += 1



