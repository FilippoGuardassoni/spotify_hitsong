![](https://github.com/FilippoGuardassoni/spotify_hitsong/blob/main/img/headerheader.jpeg)

# Song Hit Prediction using Statistical Learning

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pragyy/datascience-readme-template?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/FilippoGuardassoni/spotify_hitsong)
![GitHub pull requests](https://img.shields.io/github/issues-pr/FilippoGuardassoni/spotify_hitsong)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![contributors](https://img.shields.io/github/contributors/FilippoGuardassoni/spotify_hitsong) 
![codesize](https://img.shields.io/github/languages/code-size/FilippoGuardassoni/spotify_hitsong)

# Project Overview

The global recorded music market grew by 7.4% in 2020, the sixth consecutive year of growth, according to IFPI, the organization that represents the recorded music industry worldwide. The value of the market has been estimated to be $21.6 billion. It is hard to get a hint of how many songs an artist releases in a specific time frame but, given the fact that generally a viral song is very much less frequent than a not viral one, knowing if a specific song will be popular would give a hedge to producers and boost even more the revenues. In the current study, the problem of predicting whether a song would become a hit or non-hit was addressed using the main objective features of a song such as acousticness, tempo and danceability. A dataset of approximately 12.500 songs was created retrieving for each one the features from the Spotify Web API. After learning about the dataset through exploratory data analysis and clustering, the success of a song was able to be predicted with approximately 91% accuracy on the test set for the best model. The most successful models were the logistic regression and the tree random forest. The model that performed very poorly was the quadratic linear discriminant.

# Installation and Setup

- Tensorflow
- R/ R studio

## Codes and Resources Used
- Python 2.7 and up

## Python Packages Used
- **General Purpose:** General purpose packages like `urllib, os, request`, and many more.
- **Data Manipulation:** Packages used for handling and importing dataset such as `pandas, numpy` and others.
- **Machine Learning:** This includes packages that were used to generate the ML model such as `scikit, tensorflow`, etc.

## R Packages Used
install.packages ("ggplot2")
install.packages("gridExtra")
install.packages("dplyr")
install.packages("pastecs")
install.packages ("reshape")
install.packages("reshape2")
install.packages("factoextra")
install.packages("tidyverse")
install.packages("corrplot")
install.packages("caTools")
install.packages("pROC")
install.packages("caret")
install.packages("repr")
install.packages("glmnet")
install.packages("Rcpp")
install.packages("readr")
install.packages("randomForest")
install.packages("parameters")
install.packages("see")
install.packages("scales")
install.packages("e1071")

# Data
The song dataset contains around 12.500 records of songs with artist information and audio features.

## Source Data
### The Billboard Hot 100
The Billboard Hot 100 is the music industry standard record chart in the United States for songs since 1955, published weekly by Billboard magazine. Chart rankings are based on sales (physical and digital), radio play, and online streaming in the United States. In other words, the Billboard Hot 100 Chart remains one of the definitive ways to measure the success of a popular song. As you might have noticed, we constructed the definition of the hit variable around this context. If a song made it into the chart, it signifies that it is popular and therefore a hit. Thus, all the songs present in the charts of Billboard from each year since 1955 to 2021 were retrieved using the Billboard API. The library provided the track_name and the artist_name.

### The Million Song Dataset
A dataset of 10,000 random songs was collected from the Million Songs Dataset (MSD), a free dataset maintained by labROSA at Columbia University and EchoNest. This was narrowed down to songs released between 2007 and 2021 in order to counterbalance the skewness of the dataset. The dataset provided the artist name and song title, as well as other miscellaneous features. Finally, we removed overlapping songs. At this point, tracks were labeled 1 or 0: 1 indicating that the song was featured in the Billboard Hot 100 (between 2007- 2021) and 0 indicating otherwise.

### Spotify
Spotify API, Spotipy! to extract audio features for these songs. The Spotify API provides users with 11 audio features, and other information such as total followers and artist popularity on Spotify. This dataset was then merged with previous one to form the final one. Furthermore, the artist’s name is associated with a binary variable featuring to indicate whether an artist has collaborated with one or more other artist for a song.

## Data Acquisition
Data collection is tricky as it comes from different sources. It is all brought together in R studio:

The variables considered for the project are listed and explained below:
• hit: whether a song is a hit or not
• featuring: whether there are other artists that contributed to a song
• artist_name: the name of the artist of a song
• pop_artist: the popularity of an artist ranked from 0 to 100
• tot_followers: number of followers on spotify
• track_name: name of the song
• rel_date: date in which a song is released
• pop_track: popularity of track ranked from 0 to 100 (it can differ from our definition of hit)
• avail_mark: number of markets in which the song is present from 1 to 178
In addition, a variable for each feature is considered:
• acousticness • danceability • duration_ms • energy
• instrumentalness • liveness
• loudness
• Speechiness
• tempo
• time_signature • valence

## Data Preprocessing
Data pre-processing is carried out according to the statistical method used.

# Project structure

```bash
├── data
│   ├── billboard.csv               # data from billboard api
│   ├── millionsongs.csv            # data from millionsongs dataset
│   ├── songs.csv                   # data from spotify api
├── python
│   ├── billboardAPI.py             # API code to retrieve data from billboard 
│   ├── millionsongs.py             # code to retrieve only specific columns to use for Spotify API
│   ├── spotifyAPI.py               # API code to complete the dataset with Spotify features
│   ├── tensorflow.py               # Application of neural networks
├── r
│   ├── hitsongs.R                  # Code to pre-process data, EDA, and apply the main ML techniques
├── report
│   ├── song_hit_prediction.pdf     # written report
├── img
│   ├── headerheader.jpg            # project front image
├── LICENSE
├── README.md
└── .gitattributes,
```

# Results and evaluation
The test set was created randomly using sample.split function from R package.
The Random Forest once again performed better for the classification. The overall results are reported below:


<img width="617" alt="Screenshot 2023-05-18 at 19 45 22" src="https://github.com/FilippoGuardassoni/spotify_hitsong/assets/85356795/b420dcdd-5dbe-4170-81b7-cfe1797d705c">

All the models predicted the test set with an incredibly high consistency, showing almost no overfitting over the training dataset. Considering the training set, the same conclusions can be drawn here.
The focus was mainly on the accuracy of results, but the precision and recall for the best models are reported as well since false positive predictions may be costly when a music label invests in a song that is actually unlikely to become a hit.

<img width="475" alt="Screenshot 2023-05-18 at 19 46 59" src="https://github.com/FilippoGuardassoni/spotify_hitsong/assets/85356795/e4bfc124-b326-4150-b562-a9d66f783085">


# Future work
This study aimed to predict whether a song will be a hit or not using features from Spotify API as a base. The first limitations come in play here. The definition of what a hit is inherently has constraints. This affects the composition of the dataset as well. A broader definition of hit could impact positively the research. In addition, the features are limited to the ones that Spotify API makes available. Other features for example of sound can be inserted such as bit depth and amplitude. Feature extraction and spectrogram analysis from the song audio files could be another successful approach. Subjective preferences, seasonality and time period might be taken in account as well. As instance, certain songs or genre such as Latin music are more likely to become hits in summer than in winter. Furthermore, particular characteristics in songs that made songs famous in the past are not a good guarantee for the future. While rock music dominated in the decades before 2000, nowadays electronic music took hold. Lyrics of a song could be considered to see if there are recurrencies, patterns or words that boost listening.

# Acknowledgments/References
See report/song_hit_prediction.pdf for references.

# License
For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).
