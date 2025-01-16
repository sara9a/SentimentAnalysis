import string
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

df = pd.read_csv("sampled_data.csv")
df = df[['Id', 'Score', 'Text']]

##Converting to Lowercase
df['Clean Text'] = df['Text'].str.lower()
print(df.head(5))

##Removing Punctuations
def rem_punctuation(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('', '', punctuations))
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_punctuation(x))
##print(df.head(5))

##Removing Stopwords
stopwords.words('english')
list_stopwords = set(stopwords.words('english'))
def rem_stopwords(text):
    return " ".join([word for word in text.split() if word not in list_stopwords])
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_stopwords(x))
##print(df.head(15))

## Removing Special Characters ##
def rem_SpecialChars(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text
df['Clean Text'] = df['Clean Text'].apply(lambda x: rem_SpecialChars(x))
print(df.head(15))

## Next Step: Stemming / Lemmatization