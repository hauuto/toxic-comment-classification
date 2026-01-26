import os
import argparse
import warnings
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import joblib


def load_data(path: str, id_col: str, text_col: str, label_col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(path)
    for c in [id_col, text_col, label_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {path}")
    return df[id_col], df[text_col].astype(str), df[label_col].astype(str)


def train_and_eval(texts: pd.Series, labels: pd.Series, model_type: str) -> Tuple[Pipeline, str]:
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, lowercase=True)

    if model_type == "svm":
        clf = LinearSVC()
        pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    elif model_type == "dt":
        clf = DecisionTreeClassifier(random_state=42)
        pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    else:
        raise ValueError("model_type must be 'svm' or 'dt'")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred)
    return pipe, report


def main():
    parser = argparse.ArgumentParser(description="TF-IDF text classification (SVM/DecisionTree)")
    parser.add_argument("--input", default=os.path.join("reports", "labeled_data.csv"), help="Input CSV with id,text,label")
    parser.add_argument("--id-col", default="id", help="ID column name (default: id)")
    parser.add_argument("--text-col", default="text", help="Text column name (default: text)")
    parser.add_argument("--label-col", default="label", help="Label column name (default: label)")
    parser.add_argument("--model", choices=["svm", "dt"], default="svm", help="Classifier type")
    parser.add_argument("--models-dir", default=os.path.join("models"), help="Directory to save models")
    parser.add_argument("--report-out", default=os.path.join("reports", "classification_report.txt"), help="Path to write classification report")
    parser.add_argument("--dry-run", action="store_true", help="Skip training; just vectorize and summarize")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)

    ids, texts, labels = load_data(args.input, args.id_col, args.text_col, args.label_col)

    if labels.nunique() < 2:
        warnings.warn("Label column has fewer than 2 classes; training cannot proceed.")
        if args.dry_run:
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, lowercase=True)
            X = vec.fit_transform(texts)
            print(f"[DRY] Vectorized {X.shape[0]} docs with {X.shape[1]} features.")
            return
        else:
            return

    if args.dry_run:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, lowercase=True)
        X = vec.fit_transform(texts)
        print(f"[DRY] Vectorized {X.shape[0]} docs with {X.shape[1]} features.")
        return

    pipe, report = train_and_eval(texts, labels, args.model)
    model_path = os.path.join(args.models_dir, f"tfidf_{args.model}.pkl")
    joblib.dump(pipe, model_path)
    print(f"[OK] Saved model: {model_path}")

    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Wrote classification report: {args.report_out}")


if __name__ == "__main__":
    main()
