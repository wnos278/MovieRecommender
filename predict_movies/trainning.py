from pandas import DataFrame, read_csv
import pickle
# import matplotlib.pyplot as plt
# import matplotlib
import pandas as pd
import sys

#Đếm số lượng phim thuộc thể loại
def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in liste_keywords: 
            if pd.notnull(s): keyword_count[s] += 1
    # convert the dictionary in a list to sort the keywords  by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    # print(keyword_occurences)
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    # print(keyword_occurences)
    return keyword_occurences, keyword_count

# Load data
# Read Movies csv File
moviesFP = "E:\\20201\\ai\\data\\small\\movies.csv"
moviesDF = pd.read_csv(moviesFP)
# Read Ratings CSV File
ratingsFP = "E:\\20201\\ai\\data\\small\\ratings.csv"
ratingsDF = pd.read_csv(ratingsFP)
# Read Tasg CSV File
tagsFP = "E:\\20201\\ai\\data\\small\\tags.csv"
tagsDF = pd.read_csv(tagsFP)
tempDF = pd.merge(ratingsDF,tagsDF,on=['userId','movieId'],how='left')
mergedDF = pd.merge(tempDF,moviesDF,on=['movieId'],how='left')
# Data Rows with Null Values
mergedDF.isna().sum()

#here we  make census of the genres:
genre_labels = set()
for s in mergedDF['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))

#counting how many times each of genres occur:
keyword_occurences, dum = count_word(mergedDF, 'genres', genre_labels)

## New Notebook block
tempDF = mergedDF

for (key,cnt) in keyword_occurences:
    tempDF.loc[tempDF['genres'].str.contains(key), key] = 1
    print(tempDF)
    tempDF[key] = tempDF[key].fillna(0)

tempDF = tempDF.drop(columns=['title','genres','timestamp_y'])
tempDF['tag'] = tempDF['tag'].fillna('')


# Processing
# Use tfidf for creating feature vectors tags
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(tempDF['tag'])
ftr = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
tempDF = tempDF.drop(columns=['tag'])
# Concat Features data to actual data

ftrDF = pd.concat([tempDF, ftr], axis=1)

## Model:

# Run K-means Clustering - with clusters count as 0
# 0 because of number of genres

from sklearn.cluster import KMeans
# Initializing KMeans

kmeans = KMeans(n_clusters=50)
# Fitting with inputs
kmeans = kmeans.fit(ftrDF)
# Predicting the clusters
labels = kmeans.predict(ftrDF)
# Getting the cluster centers
C = kmeans.cluster_centers_

## Save Model
pickle.dump(kmeans, open("movies-prediction.pkl", "wb"))

## Fit_predict
clustersvdf=kmeans.fit_predict(ftrDF)

# Top 10 Movies in Cluster 0

mergedDF.iloc[list(np.where(clustersvdf==0))[0]]['title'].unique()[:10]