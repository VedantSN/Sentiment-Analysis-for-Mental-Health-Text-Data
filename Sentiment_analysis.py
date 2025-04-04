# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from PIL import Image
import pytesseract
import nltk
from wordcloud import WordCloud
from collections import Counter
import io
import spacy
from multiprocessing import Pool, cpu_count

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Original NLTK preprocessing
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Enhanced preprocessing with spaCy
def enhanced_preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Process with spaCy
    doc = nlp(text)
    
    # Combine spaCy and NLTK for optimal results
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and len(token.text.strip()) > 0:
            # Use spaCy's lemma but fall back to NLTK for pronouns
            lemma = token.lemma_ if token.lemma_ != '-PRON-' else lemmatizer.lemmatize(token.text)
            tokens.append(lemma)
    
    return ' '.join(tokens)

def preprocess_texts(texts):
    return [enhanced_preprocess_text(text) for text in texts]

# Function for parallel processing
def parallel_preprocessing(df, column_name):
    num_cores = cpu_count()
    df_split = np.array_split(df, num_cores)
    
    # Create a multiprocessing Pool
    with Pool(num_cores) as pool:
        # Preprocess the text in parallel
        results = pool.map(preprocess_texts, [batch[column_name].tolist() for batch in df_split])
    
    # Combine the results
    df['cleaned_statement'] = [item for sublist in results for item in sublist]
    return df

# OCR function to extract text from images
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Function to generate word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    return fig

# Function to extract mental health entities
def extract_mental_health_entities(text):
    doc = nlp(text)
    entities = []
    
    # Mental health related terms
    mental_health_terms = ["anxiety", "depression", "stress", "worried", "sad", 
                          "anxious", "panic", "fear", "mood", "emotion"]
    
    for token in doc:
        if token.lemma_.lower() in mental_health_terms:
            entities.append((token.text, "MENTAL_CONDITION"))
    
    return entities

# Streamlit App Title
st.title("Mental Health Text Classification")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Upload CSV", "Image Input", "Text Input"])

with tab1:
    # Upload File
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="csv_uploader")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, index_col=0)
        # Limit the dataset to 5000 entries
        # if len(df) > 5000:
        #     df = df.sample(n=5000, random_state=42)

        # Display data info
        st.write("### Data Preview")
        st.write(df.head())

        # Check for missing values
        st.write("### Missing Values")
        st.write(df.isnull().sum())

        # Drop missing values
        df = df.dropna()
        st.write("After dropping missing values:", df.shape)

        # EDA section
        st.write("### Exploratory Data Analysis")
        st.write(df.info())
        st.write(df.describe())

        # Distribution of sentiments
        sentiment_counts = df['status'].value_counts()
        st.write("Sentiment Counts:")
        st.write(sentiment_counts)

        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', title='Distribution of Sentiments', ax=ax)
        st.pyplot(fig)

        # Calculate statement lengths
        df['statement_length'] = df['statement'].apply(len)

        # Statement length statistics before outlier removal
        st.write("### Statement Length Statistics (Before Outlier Removal)")
        st.write(df['statement_length'].describe())

        # Distribution of statement lengths before outlier removal
        fig, ax = plt.subplots()
        df['statement_length'].hist(bins=100, ax=ax)
        ax.set_title('Distribution of Statement Lengths (Before Outliers Removal)')
        ax.set_xlabel('Length of Statements')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Compare statement lengths across categories using a boxplot before outlier removal
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x='status', y='statement_length', palette='coolwarm', ax=ax)
        ax.set_title('Statement Lengths by Mental Health Category (Before Outliers Removal)')
        ax.set_xlabel('Mental Health Category')
        ax.set_ylabel('Statement Length')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Violin plot before outlier removal
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=df, x='status', y='statement_length', palette='cool', split=True, inner="quartile", ax=ax)
        ax.set_title('Violin Plot of Statement Lengths Split by Category (Before Outliers Removal)')
        ax.set_xlabel('Mental Health Category')
        ax.set_ylabel('Statement Length')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Remove outliers based on IQR
        Q1 = df['statement_length'].quantile(0.25)
        Q3 = df['statement_length'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[(df['statement_length'] >= lower_bound) & (df['statement_length'] <= upper_bound)]
        
        st.write(f"Original dataset size: {len(df)} rows")
        st.write(f"After outlier removal: {len(filtered_df)} rows")
        st.write(f"Removed {len(df) - len(filtered_df)} outliers")

        # Visualizations after outlier removal
        st.write("### Visualizations After Outlier Removal")
        
        # Plot the distribution of statement lengths without outliers
        fig, ax = plt.subplots()
        filtered_df['statement_length'].hist(bins=100, ax=ax)
        ax.set_title('Distribution of Statement Lengths (After Outliers Removal)')
        ax.set_xlabel('Length of Statements')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Understand the frequency of each mental health category
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=filtered_df, x='status', palette='Set2', ax=ax)
        ax.set_title('Distribution of Mental Health Categories')
        ax.set_xlabel('Mental Health Category')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Compare statement lengths across categories using a boxplot after outlier removal
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=filtered_df, x='status', y='statement_length', palette='coolwarm', ax=ax)
        ax.set_title('Statement Lengths by Mental Health Category (After Outlier Removal)')
        ax.set_xlabel('Mental Health Category')
        ax.set_ylabel('Statement Length')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Violin plot after outlier removal
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=filtered_df, x='status', y='statement_length', palette='cool', inner="quartile", ax=ax)
        ax.set_title('Violin Plot of Statement Lengths by Mental Health Category (After Outlier Removal)')
        ax.set_xlabel('Mental Health Category')
        ax.set_ylabel('Statement Length')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Show the distribution of statement lengths by category
        fig, ax = plt.subplots(figsize=(8, 6))
        for category in filtered_df['status'].unique():
            sns.kdeplot(filtered_df[filtered_df['status'] == category]['statement_length'], label=category, ax=ax)
        ax.set_title('Density Plot of Statement Lengths by Category')
        ax.set_xlabel('Statement Length')
        ax.set_ylabel('Density')
        ax.legend()
        st.pyplot(fig)

        # Visualize word frequency across categories using a heatmap
        vectorizer = CountVectorizer(max_features=20, stop_words='english')
        X = vectorizer.fit_transform(df['statement'])
        word_freq_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        word_freq_df['status'] = df['status']

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(word_freq_df.groupby('status').mean(), cmap='YlGnBu', annot=True, ax=ax)
        ax.set_title('Heatmap of Average Word Frequency by Mental Health Category')
        st.pyplot(fig)

        # Word clouds for each category
        st.write("### Word Clouds by Category")
        for status in filtered_df['status'].unique():
            status_text = ' '.join(filtered_df[filtered_df['status'] == status]['statement'])
            fig = generate_word_cloud(status_text, title=f'Word Cloud for {status}')
            st.pyplot(fig)

        # Add a progress bar for feedback
        progress_bar = st.progress(0)

        # Choose preprocessing method
        preprocessing_method = st.radio(
            "Select preprocessing method:",
            ("NLTK (Original)", "spaCy Enhanced", "Combined (NLTK + spaCy)")
        )

        # Add a confirm button
        confirm_preprocessing = st.button("Confirm Preprocessing Method")

        # Only proceed with preprocessing when the confirm button is clicked
        if confirm_preprocessing:
            # Preprocessing
            @st.cache_data
            def preprocess_statements(df, method):
                if method == "NLTK (Original)":
                    df['cleaned_statement'] = df['statement'].apply(preprocess_text)
                elif method == "spaCy Enhanced":
                    st.info("Using spaCy for preprocessing (may take longer but provides better linguistic analysis)")
                    df = parallel_preprocessing(df, 'statement')
                else:  # Combined
                    st.info("Using combined NLTK and spaCy preprocessing")
                    df['cleaned_statement'] = df['statement'].apply(enhanced_preprocess_text)
                return df
            filtered_df = preprocess_statements(filtered_df, preprocessing_method)
            progress_bar.progress(30)

            # Train-Test Split
            X = filtered_df['cleaned_statement']
            y = filtered_df['status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # TF-IDF Vectorization
            @st.cache_data
            def vectorize_data(X_train, X_test):
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_test_tfidf = vectorizer.transform(X_test)
                return vectorizer, X_train_tfidf, X_test_tfidf

            vectorizer, X_train_tfidf, X_test_tfidf = vectorize_data(X_train, X_test)
            progress_bar.progress(60)

            # SMOTE for Balancing
            smote = SMOTE(random_state=0)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
            progress_bar.progress(80)

            # Train SVM Model
            svm_model = SVC(kernel='linear', C=1.0, probability=True)
            svm_model.fit(X_train_resampled, y_train_resampled)
            progress_bar.progress(100)

            # Classification Report
            y_pred_svm = svm_model.predict(X_test_tfidf)
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred_svm))

            # Save model and vectorizer in session state
            st.session_state['model'] = svm_model
            st.session_state['vectorizer'] = vectorizer
            st.session_state['classes'] = svm_model.classes_
            st.session_state['preprocessing_method'] = preprocessing_method

            st.success(f"Model trained using {preprocessing_method} preprocessing method!")
        else :
            st.info("Please select a preprocessing method and click 'Confirm Preprocessing Method' to proceed with training.")

with tab2:
    st.write("### Upload an Image with Text")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract text from the image
        if st.button("Extract Text and Classify"):
            with st.spinner("Extracting text from image..."):
                extracted_text = extract_text_from_image(image)

                if extracted_text.strip():
                    st.write("### Extracted Text:")
                    st.write(extracted_text)

                    # Check if model is trained
                    if 'model' in st.session_state and 'vectorizer' in st.session_state:
                        # Process and classify the extracted text
                        if st.session_state.get('preprocessing_method') == "spaCy Enhanced":
                            processed_text = enhanced_preprocess_text(extracted_text)
                        elif st.session_state.get('preprocessing_method') == "Combined (NLTK + spaCy)":
                            processed_text = enhanced_preprocess_text(extracted_text)
                        else:  # NLTK (Original)
                            processed_text = preprocess_text(extracted_text)

                            
                        vectorized_text = st.session_state['vectorizer'].transform([processed_text])
                        prediction = st.session_state['model'].predict(vectorized_text)[0]
                        probabilities = st.session_state['model'].predict_proba(vectorized_text)[0]

                        st.write("### Classification Results:")
                        st.write(f"*Predicted Category:* {prediction}")
                        st.write("Confidence Scores:", dict(zip(st.session_state['classes'], probabilities)))
                        
                        # Extract mental health entities
                        entities = extract_mental_health_entities(extracted_text)
                        if entities:
                            st.write("### Mental Health Entities Detected:")
                            for entity, label in entities:
                                st.write(f"- {entity} ({label})")
                        # Word cloud visualization for extracted text
                        if len(extracted_text.split()) > 3:
                            st.write("### Word Cloud:")
                            fig = generate_word_cloud(extracted_text, "Extracted Text Word Cloud")
                            st.pyplot(fig)

                    else:
                        st.warning("Please upload a CSV file in the 'Upload CSV' tab first to train the model.")
                else:
                    st.error("No text could be extracted from the image. Please try another image.")

with tab3:
    st.write("### Predict Mental Health Category")
    user_input = st.text_area("Enter a statement:")
    if st.button("Classify", key="classify_text"):
        if user_input:
            if 'model' in st.session_state and 'vectorizer' in st.session_state:
                # Use the same preprocessing method as training
                if st.session_state.get('preprocessing_method') == "spaCy Enhanced":
                    processed_text = enhanced_preprocess_text(user_input)
                elif st.session_state.get('preprocessing_method') == "Combined (NLTK + spaCy)":
                    processed_text = enhanced_preprocess_text(user_input)
                else:  # NLTK (Original)
                    processed_text = preprocess_text(user_input)
                    
                vectorized_text = st.session_state['vectorizer'].transform([processed_text])
                prediction = st.session_state['model'].predict(vectorized_text)[0]
                probabilities = st.session_state['model'].predict_proba(vectorized_text)[0]

                st.write(f"*Predicted Category:* {prediction}")
                st.write("Confidence Scores:", dict(zip(st.session_state['classes'], probabilities)))
                
                # Extract mental health entities
                entities = extract_mental_health_entities(user_input)
                if entities:
                    st.write("### Mental Health Entities Detected:")
                    for entity, label in entities:
                        st.write(f"- {entity} ({label})")
                        # Word cloud visualization for user input
                if len(user_input.split()) > 3:
                    st.write("### Word Cloud:")
                    fig = generate_word_cloud(user_input, "Input Text Word Cloud")
                    st.pyplot(fig)

            else:
                st.warning("Please upload a CSV file in the 'Upload CSV' tab first to train the model.")
