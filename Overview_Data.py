import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt

df = pd.read_csv("sampled_data.csv")
df = df[['Id', 'Score', 'Text']]
print(df.head(5))
print(df['Score'].value_counts())

ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Total Count of Ratings by Score',
          figsize=(10, 5))
ax.set_xlabel('Product Rating')
plt.show()
