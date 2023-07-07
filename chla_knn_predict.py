import streamlit as st
import pandas as pd
import pickle
import numpy as np

#load data 
df = pd.read_csv("https://github.com/ArinB/MSBA-CA-Data/raw/main/CA05/movies_recommendation_data.csv")

#drop 'label' column
df = df.drop('Label',axis =1)

#load pickle file 
def load_model():
    with open('knn_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

knn = data["model"]
IMDB_Rating = data["IMDB_Rating"]
Biography = data["Biography"]
Drama = data["Drama"]
Thriller = data["Thriller"]
Comedy = data["Comedy"]
Crime = data["Crime"]
Mystery = data["Mystery"]
History = data["History"]

def show_predict_page():
    st.title("Movie Recommender")

    st.write("Need Information to Recommend Movies")

    # select number for each feature - IMDB Rating
    IMDB_Rating = st.number_input('Input a number for IMDB Rating range [1.0 - 10.0]', min_value=1.0, max_value=10.0, step=0.1)
    st.write('You selected the number: ', IMDB_Rating)

    # select yes/no for genre 
    Biography = st.slider("Biography: YES = 1, NO = 0", 0, 1)
    Drama = st.slider("Drama: YES = 1, NO = 0", 0, 1)
    Thriller = st.slider("Thriller: YES = 1, NO = 0", 0, 1)
    Comedy = st.slider("Comedy: YES = 1, NO = 0", 0, 1)
    Crime = st.slider("Crime: YES = 1, NO = 0", 0, 1)
    Mystery = st.slider("Mystery: YES = 1, NO = 0", 0, 1)
    History = st.slider("History: YES = 1, NO = 0", 0, 1)

    # button to predict 
    ok = st.button("Click to See Your Movie Recommendation")
    if ok:
        x = np.array([[IMDB_Rating,Biography,Drama,Thriller,Comedy,Crime,Mystery,History]])

        distances, indices = knn.kneighbors(x)

        # for loop to print out the movie name
        for movie_title in indices[0]:
            st.subheader(f"Recommend Movies: {df.iloc[movie_title]['Movie Name']}")

import streamlit
from chla_knn_predict import show_predict_page

show_predict_page()
