Here’s a more polished, GitHub‑ready `README.md` with badges, clearer formatting, and actionable links:

````markdown
# 🎬 Sentiment Analysis of Movie Reviews

> 🚀 End‑to‑end pipeline for binary sentiment classification (positive vs. negative) on IMDb movie reviews: text cleaning, TF‑IDF and n‑gram features, Logistic Regression & Naive Bayes modeling, plus rich visualizations.


## 🗃️ Dataset

IMDb Dataset of 50K Movie Reviews from Kaggle:  
- **Columns**:  
  - `review` (string) – raw movie review text  
  - `sentiment` (string) – `positive` or `negative`

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Create & activate virtualenv
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
````

---

## 💻 Usage

1. Launch the notebook:

   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```
2. Execute sections to:

   * **Load & explore** raw data
   * **Clean & preprocess** text
   * **Extract features** (TF‑IDF & n‑grams)
   * **Train models** (Logistic Regression & Naive Bayes)
   * **Evaluate** performance
   * **Visualize** results

---

## 🧹 Preprocessing

* **Lowercase** all text
* **Strip HTML tags** (`<.*?>`)
* **Remove non‑alphabetic** characters (`[^a-z\s]`)
* **Filter stopwords** with NLTK’s English list plus domain‑specific terms

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(clean_text)
```

---

## 🔢 Feature Extraction

1. **TF‑IDF** (top 5,000 terms)

   ```python
   vectorizer = TfidfVectorizer(max_features=5000)
   X = vectorizer.fit_transform(df['clean_review'])
   ```
2. **Count Vectorizer** for **1‑grams & 2‑grams** (top 1,000)

   ```python
   vectorizer = CountVectorizer(
       ngram_range=(1,2),
       max_features=1000,
       stop_words=all_stop
   )
   pos_mat = vectorizer.fit_transform(positive_reviews)
   neg_mat = vectorizer.transform(negative_reviews)
   ```

---

## 🤖 Modeling

```python
# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
```

---

## 📊 Visualization

* **Confusion Matrix** (Seaborn heatmap)
* **Top 20 Positive/Negative Words** (annotated barplots)
* **Word Cloud** (positive reviews)
* **Comparative Bar Chart** (shared n‑grams)

---

## 📈 Results

| Model                   | Accuracy |
| ----------------------- | -------- |
| Logistic Regression     | 0.89     |
| Multinomial Naive Bayes | 0.87     |

> See full classification report & confusion matrix in the notebook.

---

## 📦 Requirements

```text
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
jupyter
```

Install:

```bash
pip install -r requirements.txt
```

---

## 📜 License

Distributed under the [MIT License](LICENSE).
Feel free to fork, improve, and submit PRs!

---

*Happy Coding!* 🚀

```
::contentReference[oaicite:0]{index=0}
```
