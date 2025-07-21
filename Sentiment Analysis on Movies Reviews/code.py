import pandas as pd 
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.head()

print(df.info())
print(df['sentiment'].value_counts())
import re
from nltk.corpus import stopwords


import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])

y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
base_stop = set(stopwords.words('english'))

domain_stop = {'movie','film','movies','films'}
all_stop = list(base_stop.union(domain_stop))


positive_reviews = df[df['sentiment'] == 'positive']['clean_review']
negative_reviews = df[df['sentiment'] == 'negative']['clean_review']

vectorizer = CountVectorizer(ngram_range=(1,2),
                             max_features=1000,
                             stop_words=all_stop)

pos_mat = vectorizer.fit_transform(positive_reviews)
neg_mat = vectorizer.transform(negative_reviews)

pos_freq = np.sum(pos_mat.toarray(), axis=0)
neg_freq = np.sum(neg_mat.toarray(), axis=0)

freq_df = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'pos': pos_freq,
    'neg': neg_freq
})

top_pos = freq_df.sort_values('pos', ascending=False).head(20)
top_neg = freq_df.sort_values('neg', ascending=False).head(20)


plt.figure(figsize=(10,6))
sns.barplot(x='pos', y='word', data=top_pos, palette='Blues_d')
for i, v in enumerate(top_pos['pos']):
    plt.text(v + 50, i, str(v), va='center')
plt.title("Top 20 Positive Words")
plt.xlabel("Frequency")
plt.ylabel("")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x='neg', y='word', data=top_neg, palette='Reds_d')
for i, v in enumerate(top_neg['neg']):
    plt.text(v + 50, i, str(v), va='center')
plt.title("Top 20 Negative Words")
plt.xlabel("Frequency")
plt.ylabel("")
plt.tight_layout()
plt.show()

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

from sklearn.metrics import accuracy_score

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))