# Movie Review Sentiment Analysis
### Natural Language Processing | Binary Text Classification

**Dataset:** IMDB Dataset of 50,000 Movie Reviews  
**Best Model:** Logistic Regression — 90.6% Accuracy | ROC-AUC 0.968  
**Tech Stack:** Python, scikit-learn, NLTK, XGBoost, Matplotlib, Seaborn

---

## Problem Statement

Online platforms receive millions of movie reviews daily. Manually categorizing them as positive or negative is not scalable. This project builds an automated NLP pipeline that classifies IMDB movie reviews into positive or negative sentiment with over 90% accuracy.

---

## Project Structure

```
Movie_Sentiment_Analysis/
│
├── Movie_Sentiment_Analysis.ipynb                 # Main notebook
├── IMDB Dataset.csv                               # Source dataset (Kaggle)
├── IMDB_Movie_Sentiment_Analysis_Visualizations                 
└── README.md
```

---

## Dataset

- **Source:** [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 reviews 
- **Features:** Raw review text + sentiment label

---

## Pipeline Overview

```
Raw Review Text
      |
      v
Text Preprocessing
  - HTML tag removal
  - Special character removal
  - Lowercasing
  - Negation-aware stopword removal (kept: not, never, no, n't)
  - Negation window tagging  (e.g. "not good" -> "NOT_good")
  - Porter Stemming
      |
      v
Feature Extraction
  - TF-IDF Vectorizer
  - Unigrams + Bigrams
  - 45,000 features
      |
      v
Model Training (with GridSearchCV)
  - Multinomial Naive Bayes
  - Logistic Regression
  - LinearSVC
  - XGBoost
      |
      v
Evaluation
  - Accuracy, Confusion Matrix
  - ROC-AUC Score, ROC Curve
  - 5-Fold Cross Validation
  - Feature Importance (LR Coefficients)
```

---

## Results

| Model | Test Accuracy | ROC-AUC |
|-------|--------------|---------|
| Naive Bayes | ~85% | — |
| Logistic Regression | **90.6%** | **0.968** |
| LinearSVC | ~90% | — |
| XGBoost | ~82% | — |

**Confusion Matrix (Logistic Regression on 9,917 test samples):**

|  | Predicted Negative | Predicted Positive |
|--|---|---|
| Actual Negative | 4431 (TN) | 509 (FP) |
| Actual Positive | 423 (FN) | 4554 (TP) |

---

## Key Technical Decisions

**1. Negation-aware preprocessing**  
Standard stopword removal deletes words like "not" and "never", which destroys the meaning of phrases like "not good" or "never boring". This project keeps negation words and implements a negation window — the next 4 words after a negation are prefixed with `NOT_`, creating distinct tokens that correctly encode negative sentiment.

**2. Bigrams in TF-IDF**  
Using `ngram_range=(1,2)` captures two-word phrases like "highly recommend" or "waste time" that carry strong sentiment signals a single-word model misses.

**3. LinearSVC over SVC**  
Standard `SVC` runs in O(n²) time — unusable on 50K samples. `LinearSVC` solves the same problem in O(n) using liblinear, completing in under 60 seconds.

**4. Logistic Regression as final predictor**  
Even when another model scores slightly higher on the test set, Logistic Regression is used for final predictions because it is the most robust to short, unseen input text and produces interpretable probability scores.

---

## Known Limitation

Rule-based negation handling fails for double negatives such as "not bad" (which means "good" in English). Solving this properly requires context-aware models like BERT which understand sentence meaning. This is a well-known open problem in classical sentiment analysis.

---



