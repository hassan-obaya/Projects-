# 📰 Fake News Detection using NLP and Machine Learning

This project applies Natural Language Processing (NLP) techniques and machine learning to detect **fake news** articles based on their text and title content.

---

## 📂 Dataset

- **Source**: [Kaggle – Real and Fake News Dataset](https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset)
- **Columns**:
  - `title`: Headline of the article
  - `text`: Full article content
  - `label`: `FAKE` or `REAL`

---

## 🛠 Features

- Drop the unnamed index column
- Combine `title` and `text` fields into one text field
- Convert `label` to binary:  
  - `FAKE` → 0  
  - `REAL` → 1
- Text preprocessing:
  - Remove HTML tags and non-alphabetic characters
  - Lowercasing, tokenization, stopword removal, lemmatization
- Prepare clean textual input (`clean_text`) for model training

---

## 🔍 Preprocessing Example

```python
text = re.sub(r'<.*?>', '', text)
text = re.sub(r'[^a-zA-Z]', ' ', text)
text = text.lower()
tokens = nltk.word_tokenize(text)
tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]


---

## 🧪 Next Steps (Optional Model Training)

Although this notebook focuses on preprocessing, you can proceed to:

* Convert `clean_text` into TF-IDF features
* Train classifiers such as:

  * Logistic Regression
  * Naive Bayes
  * Random Forest
* Evaluate performance using accuracy, precision, recall, and F1-score

---

## 🧰 Requirements

See `requirements.txt` below:

```txt
pandas
numpy
nltk
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📦 Usage

1. Download the dataset from Kaggle and place it in the working directory.
2. Run the notebook or export it as a `.py` file.
3. Preprocessed text will be stored in the `clean_text` column.

---

## ✍️ Author

* Hassan Obaya
