import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
def load_data():
    movies = pd.read_csv('C:/Users/princ/OneDrive/Desktop/New folder/movie_recommender/recommendation/datasets/tmdb_5000_movies.csv')
    credits = pd.read_csv('C:/Users/princ/OneDrive/Desktop/New folder/movie_recommender/recommendation/datasets/tmdb_5000_credits.csv')

    
    # Merge and preprocess data as per your existing code
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Function to convert genres/keywords/cast/crew data
    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]
    
    def fetch_director(text):
        return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']
    
    def collapse(L):
        return [i.replace(" ", "") for i in L]
    
    # Preprocessing
    movies.dropna(inplace=True)
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Dropping unused columns
    new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
    new['tags'] = new['tags'].apply(lambda x: " ".join(x))
    
    return new

# Calculate similarity matrix
def calculate_similarity(new):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new['tags']).toarray()
    similarity = cosine_similarity(vector)
    return similarity

# Recommendation function

def recommend(movie, new, similarity):
    if movie not in new['title'].values:
        return None  # Return None for movie not found

    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_movies = [new.iloc[i[0]].title for i in distances[1:6]]
    
    return recommended_movies

