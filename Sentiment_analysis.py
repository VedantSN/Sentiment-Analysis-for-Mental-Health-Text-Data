import pandas as pd
import numpy as np
import string
import spacy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
import re
from sklearn.naive_bayes import BernoulliNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from multiprocessing import Pool, cpu_count

# Download necessary NLTK data files
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df=pd.read_csv('/content/Combined Data.csv',index_col=0)
df.head(10)

df.tail(10)

print(df.isnull().sum())

df = df.dropna()
print(df.isnull().sum())

df['status'].nunique()

sentiment_counts=df['status'].value_counts()
print(sentiment_counts)

df['status'].unique()

df.shape

df.info()

df.describe()

sentiment_counts.plot(kind='bar', title='Distribution of Sentiments')

# Calculate the length of each statement
df['statement_length'] = df['statement'].apply(len)

# Display basic statistics of statement lengths
print(df['statement_length'].describe())

# Plot the distribution of statement lengths
df['statement_length'].hist(bins=100)
plt.title('Distribution of Statement Lengths')
plt.xlabel('Length of Statements')
plt.ylabel('Frequency')
plt.show()

#Compare statement lengths across categories using a boxplot.
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='status', y='statement_length', palette='coolwarm')
plt.title('Statement Lengths by Mental Health Category')
plt.xlabel('Mental Health Category')
plt.ylabel('Statement Length')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x='status', y='statement_length', palette='cool', split=True, inner="quartile")
plt.title('Violin Plot of Statement Lengths Split by Category')
plt.xlabel('Mental Health Category')
plt.ylabel('Statement Length')
plt.xticks(rotation=45)
plt.show()

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['statement_length'].quantile(0.25)
Q3 = df['statement_length'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bound for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
filtered_df = df[(df['statement_length'] >= lower_bound) & (df['statement_length'] <= upper_bound)]
# Plot the distribution of statement lengths without outliers
filtered_df['statement_length'].hist(bins=100)
plt.title('Distribution of Statement Lengths (Without Outliers)')
plt.xlabel('Length of Statements')
plt.ylabel('Frequency')
plt.show()

#Understand the frequency of each mental health category.
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(data=filtered_df, x='status', palette='Set2')
plt.title('Distribution of Mental Health Categories')
plt.xlabel('Mental Health Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(data=filtered_df, x='status', y='statement_length', palette='cool', split=True, inner="quartile")
plt.title('Violin Plot of Statement Lengths Split by Category')
plt.xlabel('Mental Health Category')
plt.ylabel('Statement Length')
plt.xticks(rotation=45)
plt.show()

#Compare statement lengths across categories using a boxplot.
plt.figure(figsize=(8, 6))
sns.boxplot(data=filtered_df, x='status', y='statement_length', palette='coolwarm')
plt.title('Statement Lengths by Mental Health Category')
plt.xlabel('Mental Health Category')
plt.ylabel('Statement Length')
plt.xticks(rotation=45)
plt.show()

#Visualize word frequency across categories using a heatmap.
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=20, stop_words='english')
X = vectorizer.fit_transform(df['statement'])
word_freq_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_freq_df['status'] = df['status']

plt.figure(figsize=(12, 6))
sns.heatmap(word_freq_df.groupby('status').mean(), cmap='YlGnBu', annot=True)
plt.title('Average Word Frequency by Mental Health Category')
plt.show()

import nltk
nltk.download('punkt_tab')

#Display bar plots of the most frequent words in each category.
from collections import Counter
import nltk
nltk.download('punkt')

def plot_top_words(filtered_df, category, n=10):
    text = ' '.join(filtered_df[filtered_df['status'] == category]['statement'])
    words = nltk.word_tokenize(text.lower())
    common_words = Counter(words).most_common(n)
    words, counts = zip(*common_words)
    plt.barh(words, counts, color='skyblue')
    plt.title(f'Top {n} Words in {category}')
    plt.show()

for category in filtered_df['status'].unique():
    plot_top_words(filtered_df, category)

# Show the distribution of statement lengths by category.
plt.figure(figsize=(8, 6))
for category in filtered_df['status'].unique():
    sns.kdeplot(filtered_df[filtered_df['status'] == category]['statement_length'], label=category)
plt.title('Density Plot of Statement Lengths by Category')
plt.xlabel('Statement Length')
plt.ylabel('Density')
plt.legend()
plt.show()

#Donut chart
category_counts = filtered_df['status'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), wedgeprops=dict(width=0.4))
plt.title('Donut Chart of Mental Health Categories')
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a function to generate and display a word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generate word clouds for each status
statuses = filtered_df['status'].unique()

for status in statuses:
    status_text = ' '.join(filtered_df[filtered_df['status'] == status]['statement'])
    generate_word_cloud(status_text, title=f'Word Cloud for {status}')

# Taking a sample of the dataframe, e.g., 20,000 rows
sample_size = 20000
df_sample = filtered_df.sample(n=sample_size, random_state=1)

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

     # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Process text with spacy
    doc = nlp(text)

    # Lemmatize and remove stop words
    tokens = [token.lemma_ for token in doc if not token.is_stop]

    # Join the tokens back into a single string
    return ' '.join(tokens)

def preprocess_texts(texts):
    return [preprocess_text(text) for text in texts]

# Split the data into batches for multiprocessing
num_cores = cpu_count()
df_split = np.array_split(df_sample, num_cores)

# Create a multiprocessing Pool
with Pool(num_cores) as pool:
    # Preprocess the text in parallel
    results = pool.map(preprocess_texts, [batch['statement'].tolist() for batch in df_split])

# Combine the results
df_sample['cleaned_statement'] = [item for sublist in results for item in sublist]

# Display the first few rows of the DataFrame to confirm the changes
print(df_sample[['statement', 'cleaned_statement']].head())

df_sample.head()

# Extract features and labels
processedtext = df_sample['cleaned_statement']
sentiment = df_sample['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment,
                                                    test_size=0.3, random_state=42)

print(f'X_train size: {len(X_train)}')
print(f'X_test size: {len(X_test)}')
print(f'y_train size: {len(y_train)}')
print(f'y_test size: {len(y_test)}')

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

# Use SMOTE to oversample the training data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the parameter grid
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
}

# Initialize the model
bnb = BernoulliNB()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=bnb, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and best score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')

# Train the model with the best parameters
best_bnb = grid_search.best_estimator_
best_bnb.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred_best_bnb = best_bnb.predict(X_test)

# Evaluate the model
print("Tuned Bernoulli Naive Bayes")
print(classification_report(y_test, y_pred_best_bnb))