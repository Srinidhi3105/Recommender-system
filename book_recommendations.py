import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
books =pd.read_csv('book (1).csv',sep=';',error_bad_lines= False,encoding = "latin-1")
books.columns
books.shape
books.head(5)

books.overview
books.isnull()#there are no null values

from sklearn.feature_extraction.text import TfidVectorizer

#creating vectorizer too remove all stop words
tfidf_matrix = tfidf.fit_transform(books)  
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

#computing cosine similarity on tfidf_matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)



book_index = pd.Series(books.index,index=books['Book.Title']).drop_duplicates()


def get_book_recommendations(title,topN):
    
   
    #topN = 10
    # Getting the book index using its title 
    book_id = book[title]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[book_title]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar books
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the anime index 
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar movies and scores
    book_similar_show = pd.DataFrame(columns=["name","Score"])
    book_similar_show["name"] = anime.loc[anime_idx,"name"]
    book_similar_show["Score"] = anime_scores
    book_similar_show.reset_index(inplace=True)  
    book_similar_show.drop(["index"],axis=1,inplace=True)
    print (anime_similar_show)