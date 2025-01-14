# Music & Playlist Recommender using Machine Learning Methods
## 1. Abstract
Music recommendation systems have gained immense popularity and are used in virtually every major music streaming platform. This project presents a recommendation model for music to play after a a particular song or for a playlist based on the K-Nearest Neighbors (KNN) clustering algorithm that groups songs via audio features such as tempo, valence etc. and genres. The effectiveness of KNN clustering and the results of which are investigated. 


## 2. Introduction
It often happens that someone is playing music around friends or family and don't know which song to play next that doesn't disrupt the same tempo or candence. Recommender systems have become essential in curating personalized playlists, improving user satisfaction of major music streaming platforms. Clustering-based methods are often chosen due to the fact it offers unique advantages for grouping songs with similar characteristics and generating user-specific recommendations.

In this project, a recommendation model was built using KNN clustering, leveraging both genre data and indepth data on song features to recommend a list of songs that best fit any song you input. 

## 3. Data Description
The dataset used in this project was extracted using the Python library "Spotipy". The dataset includes about 1 million tracks with 19 features between the years 2000 and 2023. There a total of 61,445 unique artists and 82 genres contained within the dataset.

## 4. Methodology
### 4.1 Importing libraries and loading data
Data and necessary libraries were imported.

### 4.2 Exploratory Data Analysis (EDA)
#### 4.2.1 Data Description
A custom function was created to show the missing data, number of unique values, and the data type for each feature in the dataframe.
#### 4.2.2 Data Cleanup
Duplicates were removed
#### 4.2.3 Visual Insights
The following insights were generated

![newplot](https://github.com/user-attachments/assets/f1004802-161b-45ba-ac93-4f50e6488a70)
*![newplot1](https://github.com/user-attachments/assets/a6c2e6e0-07e4-4c21-bd17-39d0136e6409)
![newplot2](https://github.com/user-attachments/assets/1883b06a-1825-4b50-a86c-a6e10e1ec206)

#### 4.2.4 Generating K-means Clustering 
##### Clustering by genre

<img width="771" alt="Screenshot 2025-01-14 at 3 07 21 PM" src="https://github.com/user-attachments/assets/9f38594e-9eae-425d-8072-2e46d8cec7d9" />


##### Clustering by audio features
<img width="780" alt="Screenshot 2025-01-14 at 3 06 26 PM" src="https://github.com/user-attachments/assets/c62f2c7c-f05a-4c4c-93d8-993b7a60d274" />

Both were performed via K-Nearest Neighbors. Data points with similar genres are located close to each other, the same was observed for similar audio features


#### 4.2.5 Building the recommender model
The basis of the model was used closely clustered audio feature and song genre datapoints to determine which songs are the most similar to recommend to play next.
##### Initializing spotify API client credentials
* The spotify API client credentials were initialized allowing us to use the API client credentials to access data from Spotify's database
##### Defining functions for the recommender model
Creating a few functions that will be used in the final model
* Getting song audio features (pulling the audio features for a given track needed to eventually be used to recommend songs)
* Getting song audio information (pulling basic information such as track name, year, etc.)
* Generating the median for each audio feature (in order to recommend a list of songs based on a given one, we need to use audio features. The best way to do this is to calculate the median of the value for each feature)
* Creating a dictionary that will be the input in the final model

### Results
The model has three arguments: artist_name, track_name, and year, after inputting this information for any desired song the output is a dataframe of 15 tracks that are recommended as sounding similar. The dataframe's columns are also artist_name, track_name, and year

### Conclusion
This goal of this project was to build a recommendation model for music to play after a a particular song or for a playlist based on the K-Nearest Neighbors (KNN) clustering algorithm.
that groups songs via audio features such as tempo, valence etc. and genres. Being that datapoints were clustered relatively well by audio features and genres the model was able to accurately predict songs based on a singular given song.  
