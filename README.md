# Sentiment-Analysis-for-Mental-Health
 
A Streamlit-based application that classifies mental health-related text using NLP and machine learning techniques. This tool supports multiple input formats, performs preprocessing, and visualizes data in insightful ways to aid in understanding mental health sentiments.

## 🚀 Features

- 📂 Upload CSV files with text for batch classification
- 🖼️ Upload images containing text (OCR via pytesseract) and classify extracted text
- ⌨️ Manual text input for real-time classification
- 🧹 Multiple Preprocessing Options:
  - NLTK-based
  - spaCy-enhanced
  - Hybrid (NLTK + spaCy)
- 📊 TF-IDF Vectorization
- ⚖️ SMOTE for class imbalance handling
- 🤖 SVM Model for Classification
- ☁️ Word Cloud Generation
- 🧠 Named Entity Recognition (NER) of Mental Health Terms
- 📈 Visualizations:
  - Histograms
  - Box Plots
  - Violin Plots
  - Word Clouds

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 📺
- NLTK & spaCy 🔤
- Scikit-learn 🔧
- Imbalanced-learn ⚖️
- Pytesseract 👁️‍🗨️
- Matplotlib & Seaborn 📊

## 📁 Input Formats

### CSV Upload
Expected format:
```csv (Provided)
text
"I feel overwhelmed."
"I am happy today."
```

### Image Upload
- Upload an image (.png, .jpg) with visible text.
- OCR extracts the text and classifies the sentiment.

### Manual Text Input
- Type in any sentence or paragraph related to mental health.
- See instant results!

## 🧪 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/VedantSN/Sentiment-Analysis-for-Mental-Health-Text-Data.git
cd Sentiment-Analysis-for-Mental-Health-Text-Data

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn nltk pillow pytesseract wordcloud spacy

# 4. Download NLTK and spaCy data
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm

# 5. Run the app
streamlit run app.py
```

## 🔮 Future Improvements

- Incorporate deep learning models (e.g., BERT)
- Add user login & data history
- Store results in database
- Enhanced UI design
- Multilingual support

---

> "Mental health is just as important as physical health. Let's use AI to create awareness."
