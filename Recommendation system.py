# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 05:46:41 2019

@author: SHAJI JAMES
"""

# =============================================================================
# Content based Filtering
# =============================================================================
import pandas as pd
metadata = pd.read_csv(r'E:\CAIA\Datasets\Movies_metadata\movies_metadata.csv')

#reducing data size for easy computing
sample_metadata=metadata[:50] 

#view the dataset
sample_metadata.head() 

#the attribute overview contains the content that is to be processed for recommedation
sample_metadata['overview'].head() 

#replacing the null values with empty string
sample_metadata['overview'] = sample_metadata['overview'].fillna('')

#form tf-idf matrix
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(sample_metadata['overview'])
tfidf_matrix.shape

#find cosine similarity
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim.shape

#creating a series with title as index and the corresponding index as values
indices = pd.Series(sample_metadata.index, index=sample_metadata['title']).drop_duplicates()

#getting recommendation based on the movie 'Casino'
idx = indices['Casino']
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:6] #fetching the top5
movie_indices = [i[0] for i in sim_scores]

sample_metadata['title'].iloc[movie_indices]

# =============================================================================
# Collaberatibive filtering
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
books = pd.read_csv(r'E:\CAIA\Datasets\Books Dataset\BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns=books.columns.str.replace(' ','').str.replace('-','').str.lower()
users = pd.read_csv(r'E:\CAIA\Datasets\Books Dataset\BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns=users.columns.str.replace(' ','').str.replace('-','').str.lower()
ratings = pd.read_csv(r'E:\CAIA\Datasets\Books Dataset\BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns=ratings.columns.str.replace(' ','').str.replace('-','').str.lower()

#user based
combine_book_rating = pd.merge(ratings, books, on='isbn')
sample_book_rating=combine_book_rating[:10000]
book_rating_pivot = sample_book_rating.pivot(index = 'userid', columns = 'booktitle', values = 'bookrating').fillna(0)
book_rating_pivot.head()
book_rating_title = list(book_rating_pivot.columns)
X = book_rating_pivot.values.T

from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
matrix.shape
corr = np.corrcoef(matrix)

anal_book = book_rating_title.index('Alaska')
corr_anal_book  = corr[anal_book]
result=pd.DataFrame(corr_anal_book,index=book_rating_title,columns=['corr'])
result.sort_values('corr',ascending=False).head(6)[1:]

#item based
combine_user_rating = pd.merge(ratings, users, on='userid')
sample_user_rating=combine_user_rating[:10000]
sample_user_rating.columns
user_rating_pivot = sample_user_rating.pivot(index = 'isbn', columns = 'userid', values = 'bookrating').fillna(0)
user_rating_pivot.head()
user_rating_title = list(user_rating_pivot.columns)
Y = user_rating_pivot.values.T

from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_components=12, random_state=17)
n_matrix = SVD.fit_transform(Y)
n_matrix.shape
n_corr = np.corrcoef(n_matrix)

anal_user = user_rating_title.index(277727)
corr_anal_user  = n_corr[anal_user]
n_result=pd.DataFrame(corr_anal_user,index=user_rating_title,columns=['corr'])
n_result.sort_values('corr',ascending=False).head(6)[1:]