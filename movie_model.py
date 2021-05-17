import pandas as pd
import numpy as np
#import matplotlib as plt
import pickle
#%matplotlib inline
from sklearn.feature_extraction.text import TfidfVectorizer
#from app import Recommender
import json

class build_model():
    def __init__(self):
        self.movie = ''
        self.genre=''
        self.rating = 0.0
        
    #ratings dataset
    ratings = pd.read_csv('ratings.csv')
    ratings.head()
    ratings.drop('timestamp', inplace=True, axis=1)


    #users dataset
    users = pd.read_csv('users.csv') 
    users.head()


    #movies (items) dataset
    movies = pd.read_csv('movies1.csv')
    movies.head()
    m = movies.copy()
    m1 = m.copy()
    m1.drop('movieId', inplace = True, axis=1)
    m1.head()
    
    movie_ratings = pd.merge(movies, ratings, on='movieId', how='left')
    movie_ratings.head()

    #calculate average ratings for each movie
    avg = movie_ratings.copy()
    co = avg.groupby('title')['rating'].count()
    avg1 = avg.groupby('title')['rating'].mean()
    avg1.head(10)
    avg.to_csv('m_r.csv')
     #merge users and movie_ratings dataframes
    df= pd.merge(movie_ratings, users, on='userId')
    df.head(10)
    
    #dropping unusable columns
    df.drop(['age','sex','occupation','zip_code'], inplace=True, axis=1)

    # filter the movies data frame
    movies2 = movies[movies.movieId.isin(df)]
    m2 = m1.copy()
    m2.head()
     # map movie to id:
    Mapping_file = dict(zip(movies.movieId.tolist(), movies.title.tolist()))
    Mapping_file
    m1
    '''
    FUNCTION TO RETURN THE MOVIE DETAILS OF THE MOVIE SEARCHED
    '''
    def display_movie(title, m1=m1):
        title=title.title()
        display = m1[m1.title == title]
        
        return display
    
    print(display_movie('balto'))    
    '''
    FUNCTION TO FIND TOP 20 MOVIES
    '''
    def toptwenty(df=df, avg1=avg1):
        df3 = df.copy()  
        df3.head(10)
        list(df3)    
        avg1.head(10)
        df4 = pd.merge(df3,avg1, on='title')
        df4.head(10)
        df4 = df4.drop_duplicates()
        list(df4)
        df4.drop('userId', axis=1, inplace=True)
        df4.drop(['movieId','rating_x'], axis=1, inplace=True)
        df4.columns = ['title','genres','year','rating']
        df4 = df4.drop_duplicates()
        a =  df4.sort_values(by='rating', ascending=False, axis=0).head(100)
        return a
    
    top20 = toptwenty()
        
    print(top20)
    
    '''
    CONTENT BASED FILTERING TO RECOMMEND TOP 20 MOVIES BASED ON GENRE OF THE SEARCHED MOVIE
    '''
    #Import TfIdfVectorizer from scikit-learn
    
    
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
    
    #Replace NaN with an empty string
    m['genres'] = m['genres'].fillna('')
    
    # Check and clean NaN values
    print ("Number of movies Null values: ", max(movies.isnull().sum()))
    print ("Number of ratings Null values: ", max(ratings.isnull().sum()))
    movies.dropna(inplace=True)
    ratings.dropna(inplace=True)
    print ("Number of movies Null values: ", max(movies.isnull().sum()))
    
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(m['genres'])
    
    #Output the shape of tfidf_matrix
    tfidf_matrix.shape

    # Import linear_kernel
    from sklearn.metrics.pairwise import linear_kernel
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    #Construct a reverse map of indices and movie titles
    indices = pd.Series(m.index, index=m['title']).drop_duplicates()
    
    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(title, cosine_sim=cosine_sim, top20=top20, Mappig_file=Mapping_file, indices=indices, m1=m1):
        title = title.title()

        # Get the index of the movie that matches the title
        idx = indices[title]
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:21]
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        # Return the top 10 most similar movies
        return m1.iloc[movie_indices]  
        
#    get_recommendations('Balto')
    
    titles_list = movies.title.tolist()
    titles_list
    with open('movie_dict.json', 'w') as fp:
        json.dump(Mapping_file, fp)    
      
    p1 = pickle.dump(Mapping_file, open('map.pkl','wb'))
    p2 = pickle.dump(top20, open("model.pkl", 'wb'))
    
    
if __name__ == "__main__":
    build_model()
    









