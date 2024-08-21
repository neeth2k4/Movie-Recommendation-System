import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df1 = pd.read_csv("/kaggle/input/movie-recommendation-system/movies.csv")
print(df1.head())

df1.isna().sum()

import re

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


df1['genres_list'] = df1['genres'].str.replace('|', ' ')
df1['clean_title'] = df1['title'].apply(clean_title)

movies_data = df1[['movieId', 'clean_title', 'genres_list']]
print(movies_data.head())

df2 = pd.read_csv("/kaggle/input/movie-recommendation-system/ratings.csv")
print(df2.head())

df2.isna().sum()

ratings_data = df2.drop(['timestamp'], axis=1)
print(ratings_data.head())

combined_data = ratings_data.merge(movies_data, on='movieId')
print(combined_data.head())

vectorizer_title = TfidfVectorizer(ngram_range=(1,2))

tfidf_title = vectorizer_title.fit_transform(movies_data['clean_title'])

def search_by_title(title):
    title = clean_title(title)
    query_vec = vectorizer_title.transform([title])
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies_data.iloc[indices][::-1]
    return results

movie_results = search_by_title("Toy Story")
print(movie_results)

vectorizer_genres = TfidfVectorizer(ngram_range=(1,2))

tfidf_genres = vectorizer_genres.fit_transform(movies_data['genres_list'])

def search_similar_genres(genres):
    query_vec = vectorizer_genres.transform([genres])
    similarity = cosine_similarity(query_vec, tfidf_genres).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = movies_data.iloc[indices][::-1]
    return results

gen = 'Adventure Comedy'
print(search_similar_genres(gen))

# Recommendation

def scores_calculator(movie_id):
    #find the recommendations from users who like the same movie
    similar_users = combined_data[(combined_data['movieId']== movie_id) & (combined_data['rating']>=4)]['userId'].unique()
    similar_user_recs = combined_data[(combined_data['userId'].isin(similar_users)) & (combined_data['rating']>=4)]['movieId']
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    #print(similar_user_recs)
    
    #find the recommendations from all users who have watch the movies above
    all_users = combined_data[(combined_data['movieId'].isin(similar_user_recs.index)) & (combined_data['rating']>=4)]
    all_users_recs = all_users['movieId'].value_counts() / all_users['userId'].nunique()
    #print(all_users_recs)
    
    genres_of_selected_movie = combined_data[combined_data['movieId']==movie_id]['genres_list'].unique()
    genres_of_selected_movie = np.array2string(genres_of_selected_movie)
    movies_with_similar_genres = search_similar_genres(genres_of_selected_movie)
    
    indices = []
    for index in movies_with_similar_genres[(movies_with_similar_genres['movieId'].isin(similar_user_recs.index))]['movieId']:
        indices.append(index)

        similar_user_recs.loc[indices] = similar_user_recs.loc[indices]*1.5 

    #times a factor 0.9 to movies with similar genres and all users
    indices = []
    for index in movies_with_similar_genres[(movies_with_similar_genres['movieId'].isin(all_users_recs.index))]['movieId']:
        indices.append(index)
    all_users_recs.loc[indices] = all_users_recs.loc[indices]*0.9
    
    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ['similar', 'all']
    rec_percentages['score'] = rec_percentages['similar'] / rec_percentages['all']

    rec_percentages = rec_percentages.sort_values('score', ascending=False)
    return rec_percentages

scores_calculator(3114)


# Recommendation Systems

def recommendation_results(user_input, title=0):
    # user_input = clean_title(user_input)
    title_candidates = search_by_title(user_input)
    movie_id = title_candidates.iloc[title]['movieId']
    scores = scores_calculator(movie_id)
    results = scores.head(10).merge(movies_data, left_index=True, right_on='movieId')[['clean_title', 'score', 'genres_list']]
    resutls = results.rename(columns={'clean_title': 'title', 'genres_list': 'genres'}, inplace=True)
    return results

user_input = "Toy Story"
print("Are you looking for (please choose a number): ")
for i in range(5):
    print(i, ": ", search_by_title(user_input)['clean_title'].iloc[i])

title = 0
if int(title) in range(5):
    print("We have following recommendations: ")
    print(recommendation_results(user_input, int(title)))
else:
    print("Sorry! please try again!")
    
 
 
