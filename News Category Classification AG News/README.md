بالطبع! إليك ملفين جاهزين:
````markdown
# 📰 AG News Text Classification

This project performs multi-class classification on the **AG News** dataset using various machine learning and deep learning models. It includes data preprocessing, TF-IDF feature extraction, training models (Logistic Regression, XGBoost, Neural Network), and visualization.

---

## 📂 Dataset

- Source: [AG News Classification Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- Classes: 4 categories (World, Sports, Business, Sci/Tech)
- Files: `train.csv`, `test.csv`

---

## ⚙️ Features

- Combine title and description for full-text representation
- Text preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization with top 5000 features
- Model training using:
  - Logistic Regression
  - XGBoost
  - Feedforward Neural Network (Keras)
- Evaluation using classification report
- Word cloud visualization for each class

---

## 🛠 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/ag-news-classifier.git
cd ag-news-classifier
pip install -r requirements.txt
````

Ensure you have the AG News dataset available locally or via Kaggle API.

---

## 🚀 Usage

1. Place the dataset in the correct directory path (`/kaggle/input/ag-news-classification-dataset/` or update the path in the code).
2. Run the script (e.g., in a Jupyter Notebook or Python file):

```bash
python classify_ag_news.py
```

---

## 📊 Output

* Classification performance metrics (precision, recall, F1-score)
* Word clouds per class category
* Neural network accuracy on the test set

---

## 🧠 Models Used

| Model                  | Notes                                |
| ---------------------- | ------------------------------------ |
| Logistic Regression    | Fast, interpretable baseline         |
| XGBoost                | Gradient-boosted decision trees      |
| Neural Network (Keras) | Dense layers for non-linear learning |


## 🧪 Evaluation Metric

* **Classification Report**: Accuracy, Precision, Recall, F1-score
* **Keras NN**: Validation Accuracy

---

## ✍️ Author

* Hassan Obaya
* Contributions welcome via Pull Requests!
