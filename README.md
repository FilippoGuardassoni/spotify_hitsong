## SPOTIFY_HITSONGS
![](https://github.com/FilippoGuardassoni/spotify_hitsong/blob/main/img/headerheader.jpg)

# Frequent Pattern Mining of Old Newspapers using Apriori Algorithm

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pragyy/datascience-readme-template?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/FilippoGuardassoni/old_newspaper)
![GitHub pull requests](https://img.shields.io/github/issues-pr/FilippoGuardassoni/old_newspaper)
![GitHub](https://img.shields.io/github/license/FilippoGuardassoni/old_newspaper)
![contributors](https://img.shields.io/github/contributors/FilippoGuardassoni/old_newspaper) 
![codesize](https://img.shields.io/github/languages/code-size/FilippoGuardassoni/old_newspaper)

# Project Overview

The global recorded music market grew by 7.4% in 2020, the sixth consecutive year of growth, according to IFPI, the organization that represents the recorded music industry worldwide. The value of the market has been estimated to be $21.6 billion. It is hard to get a hint of how many songs an artist releases in a specific time frame but, given the fact that generally a viral song is very much less frequent than a not viral one, knowing if a specific song will be popular would give a hedge to producers and boost even more the revenues. In the current study, the problem of predicting whether a song would become a hit or non-hit was addressed using the main objective features of a song such as acousticness, tempo and danceability. A dataset of approximately 12.500 songs was created retrieving for each one the features from the Spotify Web API. After learning about the dataset through exploratory data analysis and clustering, the success of a song was able to be predicted with approximately 91% accuracy on the test set for the best model. The most successful models were the logistic regression and the tree random forest. The model that performed very poorly was the quadratic linear discriminant.

# Installation and Setup

- Google Colab Instance

## Codes and Resources Used
- Python 2.7 and up
- PySpark

## Python Packages Used
- **General Purpose:** General purpose packages like `urllib, os, request`, and many more.
- **Data Manipulation:** Packages used for handling and importing dataset such as `pandas, numpy` and others.
- **Data Visualization:** Include packages which were used to plot graphs in the analysis or for understanding the ML modelling such as `seaborn, matplotlib` and others.
- **Machine Learning:** This includes packages that were used to generate the ML model such as `scikit, tensorflow`, etc.

# Data

The Old_Newspaper dataset contains 16,806,041 records. It contains natural language text from various newspapers, social media posts and blog pages in multiple languages.

## Source Data & 
The dataset is a subset taken from HC Corpora which is a collection of corpora for various language (66). It is an already cleaned version of the raw data from newspaper subset of the HC corpus. The dataset contains the following variables:

- Language: Language of the text.
- Source: Newspaper from which the text is from.
- Date: Date of the article that contains the text.
- Text: Sentence/paragraph from the newspaper.

## Data Acquisition
Data collection is as simple as downloading from Kaggle through its API in Google Colab.

## Data Preprocessing
This study is conducted on only English language. The other columns are disregarded. After filtering, the resulted dataset is composed by 1.010.242 rows:

- Replace special characters and expand contractions
- Split sentences
- Remove punctuation, blank spaces and capitalization
- Remove digits
- Remove stop words
- Lemmatization
- Subsampling

# Project structure

```bash
├── code
│   ├── spark_old_newspapers_market_basket_analysis.ipynb
├── report
│   ├── old_newspaper_report.pdf
├── img
│   ├── headerheader.jpg
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
