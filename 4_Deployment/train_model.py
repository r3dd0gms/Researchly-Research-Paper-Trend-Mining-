"""Re-train the Researchly classifier and save it as researchly_model.joblib.

This reproduces notebook Section 2.3 exactly (same cleaning, same tuned settings:
TF-IDF uni+bigrams, min_df=5, max_df=0.8, Linear SVM with C=0.5, balanced class
weights, fitted on ALL de-duplicated abstracts).

Usage (from the 4_Deployment folder):
    python train_model.py
    python train_model.py --data path/to/research_abstracts.json
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import denoise

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "2_Datasets" / "research_abstracts.json"
MODEL_PATH = HERE / "researchly_model.joblib"
RANDOM_STATE = 42


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_json(path)
    df = df.dropna(subset=["Abstract", "Label"])
    df["Abstract"] = df["Abstract"].astype(str).str.strip()
    df = df[df["Abstract"].str.len() > 0]
    df = df.drop_duplicates(subset="Abstract", keep="first").reset_index(drop=True)
    df["n_tokens"] = df["Abstract"].apply(denoise).str.split().str.len()
    df = df[df["n_tokens"] >= 5].reset_index(drop=True)
    df["y"] = df["Label"].map({"classical": 0, "ai": 1})
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=denoise, ngram_range=(1, 2),
                                  min_df=5, max_df=0.8, sublinear_tf=True)),
        ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    df = load_clean(args.data)
    print(f"Training on {len(df)} abstracts "
          f"({int(df['y'].sum())} ai / {int((1 - df['y']).sum())} classical)")
    model = build_pipeline().fit(df["Abstract"], df["y"])
    joblib.dump(model, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
