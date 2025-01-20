from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_filtering(movie_id, movies, ratings, n_recommendations=5):
    # Create a TF-IDF vectorizer to convert the genre column into a matrix
    tfidf = TfidfVectorizer(stop_words='english')

    # If 'description' column doesn't exist, use 'genre' column instead
    tfidf_matrix = tfidf.fit_transform(movies['genre'])
    
    # Compute cosine similarity between the input movie and all other movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the movie that matches the movie_id
    idx = movies.index[movies['movie_id'] == movie_id].tolist()[0]
    
    # Get the pairwise similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the n most similar movies
    sim_scores = sim_scores[1:n_recommendations+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top n recommended movies
    return movies[['movie_id', 'movie_title']].iloc[movie_indices]
