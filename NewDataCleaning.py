import string
import re
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("sampled_data.csv")
df = df[['Id', 'Score', 'Text']]



##Converting to Lowercase
df['Clean Text'] = df['Text'].str.lower()
##pd.set_option('display.max_columns', None)
##pd.set_option('display.max_colwidth', None)
##print(df.head(5))

##Removing Punctuations
def rem_punctuation(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('', '', punctuations))
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_punctuation(x))
##print(df.head(5))

##Removing Stopwords
stopwords.words('english')
default_stopwords = set(stopwords.words('english'))
sentiment_words = {"not", "no", "but", "yet", "cannot", "won't", "shouldn't", "couldn't"}
custom_stopwords = default_stopwords - sentiment_words
##print("Custom Stopwords List:", custom_stopwords)


def rem_stopwords(text):
    return " ".join([word for word in text.split() if word not in custom_stopwords])
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_stopwords(x))
##print(df.head(15))

## Removing Special Characters ##
def rem_SpecialChars(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_SpecialChars(x))
#print(df.head(15))

## Next Step: Stemming / Lemmatization
ps = PorterStemmer()

def stemmer(text):
    return " ".join([ps.stem(word) for word in text.split()])

df['Stemmed Text'] = df['Clean Text'].apply(lambda x: stemmer(x))
#print(df.head(15))

################### Applying Logistical regression #########################
X = df['Stemmed Text']  # Input features (text)
y = df['Score']         # Target variable (sentiment scores)

vectorizer = TfidfVectorizer()

# Transform text into numerical features
X_vectorized = vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


