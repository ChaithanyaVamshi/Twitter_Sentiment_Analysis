# Twitter Sentiment Analysis Using Natural Language Processing (NLP) with Python 

The Ultimate Guide for Identifying Sentiments and Performing Text Analysis on Twitter Data 

![image](https://user-images.githubusercontent.com/31254745/154560304-e0e02e9a-bd36-48b5-ace3-ec2d7bab0788.png)

## 1.	Introduction

Natural Language Processing (NLP) is an emerging field and a subset of machine learning which aims to train computers to understand human languages. The most common application of NLP is Sentiment Analysis. 

In the process of NLP, we aim to prepare a textual dataset to build a vocabulary for text classification. 

In this project, I will walk you through the entire process of how to do Twitter Sentiment Analysis using Python.

## 2.	What is Sentiment Analysis?

Sentiment Analysis is also known as opinion mining which is one of the applications of NLP. It is a set of methods and techniques used to extract information from text or speech. In simpler terms, it involves classifying a piece of text as positive, negative or neutral.

Twitter is one of those social media platforms where people are free to share their opinions. We mostly see negative opinions on Twitter. So, we should continue to analyse the sentiments to find the type of people who are spreading hate and negativity.

![image](https://user-images.githubusercontent.com/31254745/154560661-e90c4ea0-5a33-46aa-956f-5f3eaf97b051.png)

## 3.	Problem Statement

The objective of this task is we are given a data set of tweets that are labelled as positive and negative. Using these labelled data, we need to train a Machine learning model using Python to predict the sentiment (positive or negative) of new tweets.

## 4.	Tweets Pre-processing and Cleaning

The pre-processing of the text data is an essential step as it makes it easier to extract information from the text and apply machine learning algorithms to it.  
The objective of this step is to Inspect data and clean noise that is irrelevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms that don’t carry much weightage in context to the text.

Data Pre-processing is divided into two parts:

- Data Inspection
- Data Cleaning 

## 5.	Data Exploration and Visualization from Tweets

Exploring and visualizing data, no matter whether it's text or any other data is an essential step in gaining insights. we must think and ask questions related to the data in hand. 

1.	The common words used in the tweets: Word Cloud

![image](https://user-images.githubusercontent.com/31254745/154561074-22f7955f-2986-41fd-9948-c3d01f435767.png)

2. Words in Positive Tweets: Word Cloud

![image](https://user-images.githubusercontent.com/31254745/154561151-ea06380a-d10f-4084-95af-724f32fc767f.png)

3.	Words in Negative Tweets: Word Cloud
 
![image](https://user-images.githubusercontent.com/31254745/154561242-b5ed6653-771c-46f4-9347-ae8768fda75b.png)

## 6.	Extracting Features from Cleaned Tweets

To analyse pre-processed data, it needs to be converted into features. Depending upon the usage, text features can be constructed using assorted. In this project, we will be covering Bag-of-Words and TF-IDF.

## 7.	Model Building: Twitter Sentiment Analysis

We will build predictive models on the dataset using the two-feature set — Bag-of-Words and TF-IDF.

**Model Building using Bag-of-Words Features**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- SVM Classifier
- XGBoost Classifier

**Model Building using TF-IDF Features**

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- SVM Classifier
- XGBoost Classifier

**Summary of Accuracy Scores on Training Data Set**

Below is the summary table showing accuracy scores on Training Data set for different Machine learning models and feature extraction techniques. 

From all the models, the SVM classifier with TFI-IDF features achieved the best Accuracy of 94.648% on Training data and 84.029% accuracy on validation data. 

![image](https://user-images.githubusercontent.com/31254745/154561787-5a859042-06b5-4d4f-8fa8-c66096f24370.png)

## 8.	Predictions on Test Data set

Using the best model SVM classifier, we will make predictions on the test data set and save the predictions to a .csv file named test-predictions.csv.

![image](https://user-images.githubusercontent.com/31254745/154561890-0958ee65-5fc5-47be-ae9e-f579cc6aab96.png)

## 9.	Conclusion

In this project, we have learnt how to approach a Sentiment Analysis problem. We started with pre-processing and exploration of data. Then we extracted features from the cleaned text using Bag-of-Words and TF-IDF. 

Finally, we were able to apply different Machine Learning models using both the feature sets to classify the tweets and make predictions on the test data set using the best Machine learning model which outperformed the other models. 

## 11.	References

https://www.analyticsvidhya.com/blog/2021/07/understanding-natural-language-processing-a-beginners-guide/
https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/
https://thecleverprogrammer.com/2021/09/13/twitter-sentiment-analysis-using-python/











