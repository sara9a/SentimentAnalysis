import string
import re
import pandas as pd
import nltk
from nltk import WordNetLemmatizer, pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("sampled_data.csv")
df = df[['Id', 'Score', 'Text']]

# Preprocessing steps
## Convert to lowercase
df['Clean Text'] = df['Text'].str.lower()

## Removing punctuations
def rem_punctuation(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('', '', punctuations))
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_punctuation(x))

## Removing stopwords
default_stopwords = set(stopwords.words('english'))
sentiment_words = {"not", "no", "but", "yet", "cannot", "won't", "shouldn't", "couldn't"}
custom_stopwords = default_stopwords - sentiment_words

def rem_stopwords(text):
    return " ".join([word for word in text.split() if word not in custom_stopwords])
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_stopwords(x))

## Removing special characters
def rem_special_chars(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_special_chars(x))

## Stemming
ps = PorterStemmer()
def stemmer(text):
    return " ".join([ps.stem(word) for word in text.split()])
df['Stemmed Text'] = df['Clean Text'].apply(lambda x: stemmer(x))

# Logistic Regression with SMOTE and Stemming
X = df['Stemmed Text']  # Input features (text)
y = df['Score']         # Target variable (sentiment scores)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)  # Fit and transform on training data
X_test_vectorized = vectorizer.transform(X_test)       # Transform test data using the same vectorizer

# Apply SMOTE to the vectorized training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print("Accuracy (Stemming):", accuracy_score(y_test, y_pred))
print("Classification Report (Stemming):\n", classification_report(y_test, y_pred, zero_division=0))

# Lemmatization
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

def lemmatize_words(text):
    pos_text = pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])
df['Lemmatized Text'] = df['Clean Text'].apply(lambda x: lemmatize_words(x))

# Logistic Regression with SMOTE and Lemmatization
X = df['Lemmatized Text']  # Input features (text)
y = df['Score']            # Target variable (sentiment scores)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)  # Fit and transform on training data
X_test_vectorized = vectorizer.transform(X_test)       # Transform test data using the same vectorizer

# Apply SMOTE to the vectorized training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print("Accuracy (Lemmatization):", accuracy_score(y_test, y_pred))
print("Classification Report (Lemmatization):\n", classification_report(y_test, y_pred, zero_division=0))