"""Researchly - Research Papers Expert System (Streamlit web interface).

Paste an abstract and the model says whether the paper uses AI methods or classical
methods, and shows which words pushed the decision. A second tab classifies many
abstracts at once from a CSV / JSON file.

Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import preprocessing  # noqa: F401  (needed so joblib can find denoise when loading the model)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "researchly_model.joblib"
LABELS = {0: "classical", 1: "ai"}

EXAMPLES = {
    "AI example (LSTM stock forecasting)":
        "Deep neural networks and machine learning algorithms such as LSTM are trained to "
        "forecast stock market returns using large volumes of data.",
    "Classical example (New Keynesian model)":
        "A New Keynesian model with sticky prices and nominal rigidities is used to study "
        "monetary policy shocks and the Phillips curve.",
}

st.set_page_config(page_title="Researchly", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():
        # First run without a saved model: train it from the dataset in ../2_Datasets
        from train_model import DEFAULT_DATA, build_pipeline, load_clean
        df = load_clean(DEFAULT_DATA)
        model = build_pipeline().fit(df["Abstract"], df["y"])
        joblib.dump(model, MODEL_PATH)
        return model
    return joblib.load(MODEL_PATH)


def predict(model, texts):
    """Return (labels, decision scores). Positive score -> 'ai', negative -> 'classical'."""
    scores = model.decision_function(texts)
    return (scores > 0).astype(int), scores


def explain(model, text, top_n=8):
    """Terms in this abstract that pushed the Linear SVM towards 'ai' or 'classical'."""
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    row = tfidf.transform([text])
    terms = tfidf.get_feature_names_out()
    contrib = row.multiply(clf.coef_[0]).tocoo()
    df = pd.DataFrame({"term": terms[contrib.col], "contribution": contrib.data})
    df = df.sort_values("contribution")
    return df[df.contribution > 0].tail(top_n)[::-1], df[df.contribution < 0].head(top_n)


def strength(score):
    s = abs(score)
    return "borderline" if s < 0.3 else "moderate" if s < 0.8 else "strong"


model = load_model()

st.title("🔎 Researchly")
st.caption("Does this research paper use **AI** methods or **classical** methods? "
           "Paste its abstract below.")

tab_single, tab_batch, tab_about = st.tabs(["Single abstract", "Batch file", "About the model"])

# ----------------------------------------------------------------------------- single
with tab_single:
    choice = st.selectbox("Try an example (optional)", ["-"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    abstract = st.text_area("Abstract", value=default, height=200,
                            placeholder="Paste a paper abstract here...")

    if st.button("Classify", type="primary"):
        if len(abstract.split()) < 10:
            st.warning("Please paste a full abstract (at least ~10 words).")
        elif not preprocessing.denoise(abstract).strip():
            st.warning("Nothing usable left after cleaning the text. Try a longer abstract.")
        else:
            (label,), (score,) = predict(model, [abstract])
            name = LABELS[int(label)]
            colour = "#C2410C" if name == "ai" else "#1D4ED8"
            st.markdown(
                f"<div style='padding:1rem;border-radius:8px;border:2px solid {colour};'>"
                f"<span style='font-size:1.6rem;font-weight:700;color:{colour}'>"
                f"{name.upper()}</span> &nbsp; {strength(score)} signal "
                f"(decision score {score:+.2f})</div>", unsafe_allow_html=True)
            st.caption("The decision score is the Linear SVM's distance from the boundary "
                       "(0 = undecided). It is not a probability.")

            toward_ai, toward_classical = explain(model, abstract)
            c1, c2 = st.columns(2)
            c1.subheader("Pushes towards AI")
            c1.dataframe(toward_ai.round(3), hide_index=True, width="stretch")
            c2.subheader("Pushes towards classical")
            c2.dataframe(toward_classical.round(3), hide_index=True, width="stretch")
            st.caption("Terms are shown in their stemmed form (e.g. 'learn' for learning).")

# ----------------------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a **CSV** or **JSON** file with an `Abstract` column "
             "(the project's `research_abstracts.json` works).")
    up = st.file_uploader("File", type=["csv", "json"])
    if up is not None:
        try:
            data = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_json(up)
        except Exception as exc:  # malformed file
            st.error(f"Could not read the file: {exc}")
            data = None
        if data is not None:
            if "Abstract" not in data.columns:
                st.error("The file needs a column called 'Abstract'.")
            else:
                data = data.dropna(subset=["Abstract"]).copy()
                labels, scores = predict(model, data["Abstract"].astype(str))
                data["Predicted"] = [LABELS[int(i)] for i in labels]
                data["Score"] = np.round(scores, 3)
                st.write(data["Predicted"].value_counts().rename("papers").to_frame().T)
                st.dataframe(data.drop(columns=["Abstract"]).join(
                    data["Abstract"].str.slice(0, 120).rename("Abstract (start)")),
                    width="stretch")
                st.download_button("Download predictions (CSV)",
                                   data.to_csv(index=False).encode("utf-8"),
                                   "researchly_predictions.csv", "text/csv")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Model:** TF-IDF features (single words and word pairs) feeding a Linear SVM
(`C=0.5`, balanced class weights), tuned with grid search and trained on 1,125
de-duplicated abstracts (21% AI, 79% classical).

**Cleaning:** each abstract is lower-cased, stripped of URLs, DOIs, LaTeX and copyright
notices, stop-words removed and stemmed - the same `denoise()` used in the notebook.

**Expected accuracy:** about 0.86 macro-F1 in repeated cross-validation. The model is
better at recognising classical papers than AI papers, so treat borderline scores with care.

**Limits:** the training papers are mostly economics / finance research, so abstracts from
other fields may be classified less reliably.
""")
