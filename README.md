# Sentiment-Analysis-for-Mental-Health-Text-Data
 
## Overview
This project focuses on sentiment analysis of mental health-related text data using machine learning techniques. It preprocesses text data, extracts features using TF-IDF vectorization, and trains a Bernoulli Naive Bayes model optimized via GridSearchCV and SMOTE oversampling. The project also includes interactive prompts for seamless execution and visualization of insights like word clouds and sentiment distribution.

## Features
- **Data Preprocessing**: Includes tokenization, lemmatization, stopword removal, and handling outliers.
- **Feature Extraction**: Uses TF-IDF vectorization for text representation.
- **Model Training**: Implements Bernoulli Naive Bayes with hyperparameter tuning using GridSearchCV.
- **Oversampling**: Applies SMOTE to balance the dataset.
- **Visualization**: Generates word clouds, sentiment distribution plots, and mental health category comparisons.
- **Interactive Prompts**: Allows users to input text for sentiment prediction and visualize insights.

## Installation
1. Clone the repository:
2. Install required Python packages:


## Usage
1. Run the program:
2. Follow the interactive prompts to input text and view predictions or visualizations.

## Dataset
The dataset used in this project contains mental health-related statements categorized by sentiment. It is preprocessed to remove null values and outliers before analysis.

## Visualizations
The project includes several visualizations:
- Distribution of statement lengths.
- Word frequency heatmaps across categories.
- Word clouds for each mental health category.
- Violin plots and boxplots comparing text lengths by category.

## Technologies Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK, Spacy, WordCloud, Imbalanced-learn.
- **Machine Learning Techniques**: TF-IDF Vectorization, Bernoulli Naive Bayes, SMOTE Oversampling.
