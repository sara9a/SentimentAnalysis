import string
import re
from operator import pos

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
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

##Lemmatization

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "R":wordnet.ADV, "J":wordnet.ADJ}

def lemmatize_words(text):
    pos_text = pos_tag(text.split())
    return " ".join([lemmatizer(word, wordnet_map.get(pos[0], wordnet.NOUN))for word in pos_text])

df['lemmatized_text']= df['Clean Text'].apply(lambda x: lemmatize_words(x))
df.head()
