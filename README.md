# IMDb Movie Review Sentiment Analysis Using TF-IDF and Machine Learning

## Project Overview

This project develops a classical machine learning model for binary
sentiment classification of IMDb movie reviews. The model analyzes the
textual content of a movie review and predicts whether the sentiment is
**Positive** or **Negative**.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)**
for text representation and a **Linear Support Vector Machine
(LinearSVC)** for classification. The complete workflow is packaged into
a scikit-learn Pipeline and integrated into a Streamlit application for
interactive predictions.

## Dataset

The project uses the **IMDb Large Movie Review Dataset**, a benchmark
dataset for binary sentiment classification.

Dataset characteristics:

-   Total labeled reviews: 50,000
-   Training reviews: 25,000
-   Test reviews: 25,000
-   Classes: Positive and Negative
-   Task: Binary text classification
-   Input: Movie review text
-   Target: Sentiment

The dataset also contains additional unlabeled reviews, which were not
used in this supervised learning project.

## Machine Learning Workflow

**Raw Movie Review → TF-IDF → LinearSVC → Positive / Negative**

### TF-IDF Configuration

-   Lowercase conversion
-   English stop-word removal
-   Unigrams and bigrams
-   `min_df=2`
-   `max_df=0.95`
-   Sublinear term frequency

### Model

A Linear Support Vector Machine is used for sentiment classification.

Five-fold cross-validation on the training dataset selected:

``` text
C = 1.0
```

No deep learning, transformers, embeddings, PCA, or TruncatedSVD are
used.

## Model Evaluation

The final model was evaluated on the held-out IMDb test dataset.

  Metric         Score
  ----------- --------
  Accuracy      88.89%
  Precision     89.20%
  Recall        88.49%
  F1-score      88.84%

Confusion matrix:

``` text
[[11161  1339]
 [ 1439 11061]]
```

## Streamlit Application

The Streamlit application allows users to:

1.  Select a movie from a predefined list.
2.  Enter a movie review.
3.  Submit the review for analysis.
4.  Receive a **Positive** or **Negative** sentiment prediction.

The selected movie is UI context only and is not used as a machine
learning feature.

The application loads the complete scikit-learn Pipeline, containing
both TF-IDF and LinearSVC, so raw review text can be passed directly to
the model.

Because this implementation uses LinearSVC without probability
calibration, the application does not display artificial confidence or
probability percentages.

## Project Structure

``` text
IMDb-Movie-Review-Sentiment/
│
├── app.py
├── IMDb_Sentiment_Model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run Locally

``` bash
git clone <repository-url>
cd imdb-movie-review-sentiment
pip install -r requirements.txt
streamlit run app.py
```

## Model Artifact

The trained model is saved as:

``` text
IMDb_Sentiment_Model.pkl
```

The artifact contains the complete TF-IDF and LinearSVC Pipeline
required for inference.

The original IMDb dataset is not included in the repository.

## Limitations

-   Binary sentiment classification only.
-   No probability or calibrated confidence score.
-   The selected movie is not used as a prediction feature.
-   Traditional TF-IDF features may struggle with sarcasm, ambiguity,
    and complex contextual sentiment.

## References

-   IMDb Large Movie Review Dataset --- Stanford AI Lab
-   Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment
    Analysis*. Proceedings of ACL.
-   scikit-learn documentation for `TfidfVectorizer`
-   scikit-learn documentation for `LinearSVC`

## Author

**Trivikram Kambhampati**
