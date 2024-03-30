Yelp Review Sentiment Analysis

Overview

This project focuses on sentiment analysis of Yelp reviews using Natural Language Processing (NLP) techniques. The goal is to classify reviews into positive and negative  sentiments based on the review text. The project includes data preprocessing, exploratory data analysis (EDA), model development, and a simple web application for inference.

Dataset

The dataset used for this project was downloaded from Kaggle in a JSON format. It contains information such as rating stars, user IDs, and review text for user comments.

Data Preprocessing

In the yelp_review_analysis_data_preprocessing.ipynb Jupyter Notebook file, the text data was preprocessed by:

Removing stop words, numbers, and punctuation.
Normalizing the text.
A new CSV file was created containing:

Rating stars with values of 1 and 5.
Normalized text.
Text length.
Exploratory Data Analysis (EDA)

In the yelp_review_analysis_model_development.ipynb Jupyter Notebook file, EDA was conducted to analyze:

The distribution of positive (5-star) and negative (1-star) classes.
Most frequent words.
Text length distribution.
Model Development

Various models were developed and optimized to achieve the best performance, including:

Embedding model.
Long Short-Term Memory (LSTM) model.
Gated Recurrent Unit (GRU) model.
After model optimization, three models were selected, each representing the best model for the embedding model, LSTM model, and GRU model. These models were stored in joblib files.

The performance of each model was evaluated, and the embedding model was chosen as the best model based on its accuracy of 95% and loss of 21%, which were the highest accuracy and lowest loss among the three models.

Web Application

A simple web application was developed to allow users to input text and receive a sentiment analysis result as either negative or positive. The architecture of the webpage is defined in the index.html file in the template folder. The web application is created by running the server.py file (Flask).

Files

yelp_review_analysis_data_preprocessing.ipynb: Jupyter Notebook for data preprocessing.
yelp_review_analysis_model_development.ipynb: Jupyter Notebook for model development.
data/: Folder containing the original dataset and the preprocessed CSV file.
model_files/: Folder containing the joblib files for the selected models.
templates/: Folder containing the index.html file for the web application.
server.py for starting the Flask server for the web application


Dependencies

Python 3.10.13
Libraries: 

numpy: For numerical operations and array handling.

pandas: For data manipulation and analysis.

matplotlib: For creating plots and visualizations.

seaborn: For enhancing the visualizations and statistical graphics.

string: For string manipulation operations.

nltk.corpus: For accessing the NLTK (Natural Language Toolkit) corpus for stopwords.

collections.Counter: For counting the occurrences of elements in a list.

nltk.tokenize.word_tokenize: For tokenizing words in the text.

keras.preprocessing.text.Tokenizer: For tokenizing the text data.

keras.preprocessing.sequence.pad_sequences: For padding sequences to a fixed length.

sklearn.model_selection.train_test_split: For splitting the dataset into training and testing sets.

keras.models.Sequential: For creating a sequential model in Keras.

keras.layers: For adding layers to the neural network model, including Dense, Flatten, Embedding, LSTM, GRU, Dropout, and Bidirectional layers.

keras.callbacks.EarlyStopping: For implementing early stopping in model training.

keras.callbacks.TensorBoard: For visualizing model training and performance.

joblib: For saving and loading models using joblib files.


To reproduce the analysis and run the web application, follow these steps:

Download the dataset from Kaggle.
Run the yelp_review_analysis_data_preprocessing.ipynb notebook to preprocess the data.
Run the yelp_review_analysis_model_development.ipynb notebook to develop and optimize the models.
Run the server.py file to start the Flask server for the web application.
