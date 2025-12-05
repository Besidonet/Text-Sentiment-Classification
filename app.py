import streamlit as st
import pandas as pd
import numpy as np
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------------------------------
# CACHED HELPERS
# -----------------------------------------------------------

@st.cache_resource
def load_spacy_model():
    """
    Load spaCy English model once.
    Make sure you've run:
        python -m spacy download en_core_web_sm
    """
    return spacy.load("en_core_web_sm")


def preprocess_text(text, nlp):
    """
    Same preprocessing as in your notebook:
    - lowercase
    - lemmatize
    - remove stopwords
    - keep 'not'
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and token.text != "not"
    ]
    return " ".join(tokens)


@st.cache_data
def load_and_train_model(csv_path="sentiment_analysis_dataset.csv"):
    """
    Load dataset, preprocess, vectorize with TF-IDF, and train Logistic Regression.
    Returns:
        df            : original dataframe with cleaned_text
        df_filtered   : filtered dataframe (classes with >=2 samples)
        tfidf         : fitted TfidfVectorizer
        model         : trained LogisticRegression model
        metrics       : dict with accuracy, report, confusion matrix, labels
    """
    nlp = load_spacy_model()

    # -----------------------------
    # Load dataset (same as notebook)
    # -----------------------------
    df = pd.read_csv(csv_path, on_bad_lines="warn")

    # Ensure consistent column names: ['Comment', 'Sentiment']
    df.columns = ["Comment", "Sentiment"]

    # Preprocess text
    df["cleaned_text"] = df["Comment"].astype(str).apply(lambda x: preprocess_text(x, nlp))

    # Filter out sentiment classes with only one member (same as notebook)
    sentiment_counts = df["Sentiment"].value_counts()
    minority_classes = sentiment_counts[sentiment_counts < 2].index
    df_filtered = df[~df["Sentiment"].isin(minority_classes)].copy()

    # -----------------------------
    # TF-IDF Vectorization (same params)
    # -----------------------------
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = tfidf.fit_transform(df_filtered["cleaned_text"])
    y = df_filtered["Sentiment"]

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Train Logistic Regression
    # -----------------------------
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y.unique())

    metrics = {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
    }

    return df, df_filtered, tfidf, model, metrics


def predict_sentiment(text, tfidf, model, nlp):
    """
    Use same preprocessing + TF-IDF + Logistic Regression
    to predict sentiment for a new comment.
    """
    cleaned = preprocess_text(text, nlp)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction, cleaned


# -----------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Text Sentiment Classifier",
        page_icon="💬",
        layout="centered"
    )

    st.title("💬 Text Sentiment Classification App")
    st.write(
        "This app uses **spaCy**, **TF-IDF**, and **Logistic Regression** "
        "to classify comments into sentiment classes (e.g., Positive, Negative, Neutral)."
    )

    # Load model + data once
    with st.spinner("Loading model and data..."):
        df, df_filtered, tfidf, model, metrics = load_and_train_model()

    nlp = load_spacy_model()

    # -------------------------------------------------------
    # Tabs: Overview | Try it
    # -------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Dataset & Model Overview", "🧪 Try Your Own Text"])

    # --------------------- TAB 1: Overview ---------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Class Distribution (Filtered)")
        st.write(df_filtered["Sentiment"].value_counts())

        st.subheader("Model Performance")
        st.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
        st.text("Classification Report:")
        st.text(metrics["report"])

        # Confusion matrix as a simple table
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        labels = metrics["labels"]
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        st.dataframe(cm_df)

    # --------------------- TAB 2: Try it ---------------------
    with tab2:
        st.subheader("Enter a comment to classify")

        user_input = st.text_area(
            "Type your text here:",
            value="I love this video so much!",
            height=100
        )

        if st.button("Predict Sentiment"):
            if user_input.strip() == "":
                st.warning("Please type a comment first.")
            else:
                prediction, cleaned = predict_sentiment(
                    user_input, tfidf, model, nlp
                )
                st.markdown(f"**Predicted Sentiment:** `{prediction}`")
                st.markdown(f"**Preprocessed Text:** `{cleaned}`")

        st.markdown("---")
        st.markdown("Examples to try:")
        examples = [
            "I love this video so much!",
            "This is the worst thing ever.",
            "It's okay, nothing special.",
        ]
        for ex in examples:
            if st.button(f"Use example: {ex}"):
                st.session_state["example_text"] = ex

        # If an example was clicked, update the text area
        if "example_text" in st.session_state:
            st.text_area(
                "Type your text here:",
                value=st.session_state["example_text"],
                height=100,
                key="example_text_area"
            )


if __name__ == "__main__":
    main()
