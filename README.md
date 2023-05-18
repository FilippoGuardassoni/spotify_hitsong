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
- **Data Visualization:** Include packages which were used to plot graphs in the analysis or for understanding the ML modelling such as `seaborn, matplotlib` and others.
- **Machine Learning:** This includes packages that were used to generate the ML model such as `scikit, tensorflow`, etc.

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
├── README.md
└── .gitattributes,
```

# Results and evaluation
After a great number of trials, the Apriori algorithm was considered too inefficient to run with the whole dataset. In fact, these actions were taken:

1. Several values of minimum support and confidence were considered such as 0.01, 0.05, 0.001.
2. The code was run for as long as 7 hours without a final output.
3. Different best practices to optimize the code and the Spark environment were followed.

Then, a subsample of the dataset was randomly created using n = 500. Min. Support is equal to 0.01. Finally, the confidence values are calculated and printed in a dataframe in descending order.

<img width="201" alt="image" src="https://github.com/FilippoGuardassoni/old_newspaper/assets/85356795/800db652-9bc1-442b-8d56-bf105b75d1a8">

Again, the same kind of steps highlighted above were followed in the implementation of the FP-Growth algorithm. Unlike Apriori, frequent itemsets and association rules were found. The association rule table above represent the patterns that can be discerned from the dataset.

<img width="394" alt="image" src="https://github.com/FilippoGuardassoni/old_newspaper/assets/85356795/27303fd2-160e-433b-99bc-1c31b10318d0">

In conclusion, FP-Growth algorithm has revealed to the more efficient to implement when limited resources are involved (even time) to analyze a very big dataset.

# Future work
The first step to improve this analysis would be to gather additional variables to create context for the analysis, and create a basket for each context. In this case, for example, the genre of the newspaper or even of the news itself. In this way, the patterns found could have more potential meaning. Apriori and FP-Growth were implemented in this analysis, however new ones are constantly being proposed and tested such as Eclat, PCY and YAFIM.

# Acknowledgments/References
See report/old_newspaper_report.pdf for references.

# License
For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).
