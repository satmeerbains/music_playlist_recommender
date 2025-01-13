# Music Playlist Recommender using Machine Learning Methods
## 1. Abstract
Music recommendation systems have gained immense popularity and are used in virtually every major music streaming platform. This project presents a Music Playlist Recommendation model based on the K-Nearest Neighbors (KNN) clustering algorithm that groups songs via audio features such as tempo, valence etc and genres. The effectiveness of KNN clustering and the results of which are investigated. 


## 2. Introduction
How many times were you playing music around friends/family and don't know which song to play next maintains the same vibe and energy? Recommender systems have become essential in curating personalized playlists, improving user satisfaction of major music streaming platforms. Clustering-based methods were chosen due to the fact it offers unique advantages for grouping songs with similar characteristics and generating user-specific recommendations.

In this project, I propose a Music Playlist Recommender system built using KNN clustering, leveraging both genre data and song features in order to recommend a list of songs that best fit any song you input. 

## 3. Data Description
The dataset used in this project was extracted from the Spotify platform using the Python library "Spotipy". The dataset includes about 1 Million tracks with 19 features between 2000 and 2023. Also, there is a total of 61,445 unique artists and 82 genres in the data.

## 4. Methodology
### 4.1 Importing libraries and loading data
Data and necessary libraries were imported.
### 4.2 Exploratory Data Analysis (EDA)
#### 4.2.1 Data Description
A custom function was created to show the missing data, number of unique values, and the data type for each feature in our dataframe
#### 4.2.2 Data Cleanup
Duplicates were removed
#### 4.2.3 Visual Insights
The following insights were generated
* Top 5 artists based on popularity
* Top 5 most popular artists and their associated audio features
* Top 5 genres based on popularity
#### 4.2.4 Generating K-means Clustering 
Clustering by genre and audio features, as expected, we have data points with similar genres that are located close to each other.
#### 4.2.5 Building the recommender model
The basis of the model will be using closely clustered audio feature datapoints to determine which songs are the most similar to recommend to play next.
##### Initializing spotify API client credentials
* First the spotify API client credentials were intialized allowing us to use the API client credentials to access data from Spotify's database
##### Defining functions for the recommender model
Creating a few functions that will be used in the final model
* Getting song audio features (pulling the audio features for a given track needed to eventually be used to recommend songs)
* Getting song audio information (pulling basic information such as track name, year, etc)
* Generating the median for each audio feature (n order to recommend a list of songs based on a given one, we need to use audio features. The best way to do this is to caculate the median of the value for each feature)
