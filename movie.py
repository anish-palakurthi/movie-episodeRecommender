import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# taking input from user
movie_name = input("Enter favorite movie name: ")

# loading data from csv file to pandas
data = pd.read_csv("movies.csv")


# finding closest match for the movie name entered by user
list_titles = data['title'].tolist()
matches = difflib.get_close_matches(movie_name, list_titles)
closest_match = matches[0]
print("Closest match found: ", closest_match)
index = data[data.title == closest_match]['index'].values[0]


# selecting significant features
sig_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# filling NaN values with empty string
for feature in sig_features:
    data[feature] = data[feature].fillna('')

# combines the text from all the features selected into one string for vectorization
combine_features = data['genres'] + " " + data['keywords'] + " " + \
    data['tagline'] + " " + data['cast'] + \
    " " + data['director']


vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combine_features)

# calculating cosine similarity between vectors
similarity = cosine_similarity(feature_vectors)


# calculate similarity scores of all movies against the movie entered by user
similarity_score = list(enumerate(similarity[index]))
sorted_similar_movies = sorted(
    similarity_score, key=lambda x: x[1], reverse=True)

print("Movies suggested for you:\n ")
counter = 1
for movie in sorted_similar_movies[1:10]:
    index = movie[0]
    title = data[data.index == index]['title'].values[0]
    print(counter, title)
    counter += 1
