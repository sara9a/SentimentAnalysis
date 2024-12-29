# Notwendige Bibliotheken importieren
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# NLTK-Daten herunterladen
nltk.download('stopwords')
nltk.download('punkt')

# CSV-Datei laden
df = pd.read_csv("sampled_data.csv")
# 1. Bereinigung: Leere oder unvollständige Einträge entfernen
df = df.dropna(subset=['Text', 'Score'])

# 2. Text-Formatierung: Konvertiere alle Texte in Kleinbuchstaben
df['Text'] = df['Text'].str.lower()

# 3. Sonderzeichen entfernen: Entferne alle Sonderzeichen wie !, @, #, ?, etc.
import string
df['Text'] = df['Text'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))

# 4. Stopwords entfernen
from nltk.corpus import stopwords
stop_words = set(stopwords.words('german'))  # Hier kannst du 'english' für englische Texte verwenden
df['Text'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# 5. Zielvariable (Sentiment) erstellen: Binär-Sentiment-Label basierend auf Score
df['Sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)  # 1 für positiv, 0 für negativ

# 6. Text- und Zielvariable extrahieren
X = df['Text']
y = df['Sentiment']

# 7. Train-Test Split (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Vektorisierung des Textes mit TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # max_features auf eine Zahl begrenzen, um übermäßige Merkmale zu vermeiden
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 9. Modell mit Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 10. Vorhersagen machen
y_pred = model.predict(X_test_vec)

# 11. Modellbewertung
print("Modellgenauigkeit:", accuracy_score(y_test, y_pred))
print("\nKlassifikationsbericht:\n", classification_report(y_test, y_pred))

# 12. Visualisierung der Ergebnisse (z.B. eine Verteilung der Vorhersagen)
plt.figure(figsize=(6, 4))
plt.hist(y_pred, bins=2, edgecolor='black')
plt.title('Verteilung der Vorhersagen (0 = negativ, 1 = positiv)')
plt.xlabel('Vorhersage')
plt.ylabel('Anzahl der Bewertungen')
plt.xticks([0, 1], ['Negativ', 'Positiv'])
plt.show()