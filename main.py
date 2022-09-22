from operator import index
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
#TfidfVectorizer  textured data -----> numerical value
from sklearn.metrics.pairwise import cosine_similarity
#cosine_similarity -------> for finding similarity
def recommend():
    st.title("MOVIE RECOMMENDER")
    movies_data = pd.read_csv('movies.csv')
    #print(movies_data.shape)
    selected_features = ['genres','keywords','tagline','cast','director']
    #print(selected_features)
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
    #print(combined_features)
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    #print(feature_vectors)
    similarity = cosine_similarity(feature_vectors)
    #print(similarity)
    movie_name = st.text_input("Enter your movie:","batman")
   # if movie_name not in movies_data['title'].tolist():
    #    st.text("enter a movie name to start")
    #else:
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse=True)
    list_of_all_titles = movies_data['title'].tolist()
        #print(list_of_all_titles)
    
        #print(find_close_match)
    
    
        #print(index_of_the_movie)
        #print(similarity_score)
        #print(sorted_similar_movies)
    st.subheader("Movies suggested for you: \n")
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index==index]['title'].values[0]
        if(i<30):
            st.text("{}".format (title_from_index))
            i+=1   