# ITNPA – Research Papers: Trend Mining AI (Submission Package)

| Folder | Contents |
|---|---|
| `1_Documentation/` | `ITNPA_Document.docx` – project documentation for Questions 1–4 |
| `2_Datasets/` | `research_abstracts.json` (1,138 labelled abstracts) and `research_corpus/research_1.txt … research_5.txt` (five full articles) |
| `3_Notebooks/` | `ITNPA.ipynb` – the full analytics flow, Q1–Q4, with all outputs saved; `lda_topics.html` – the interactive pyLDAvis map from Q4 |
| `4_Deployment/` | Streamlit web app for the Q1 & Q2 design, with its own `README.md` for easy deployment |

## Where each question lives in the notebook

| Question | Notebook sections |
|---|---|
| Q1 – denoise, feature modelling, baseline | 1 Corpus dataframe · 1.2 Feature modelling · 1.3 Baseline and performance report |
| Q2 – four+ models, enhancement, web-interface model | 2.1 Model comparison · 2.2 Performance enhancement · 2.3 Final model for the web interface |
| Q3 – article dataframe, summaries, visualisations | 3.1–3.4 |
| Q4 – LDA topic modelling and visualisation | 4 and 4.2 |

## Re-running the notebook

```bash
pip install pandas numpy scipy scikit-learn nltk seaborn matplotlib joblib pyLDAvis jupyter
cd 3_Notebooks
jupyter notebook ITNPA.ipynb      # Run All
```

Run it from inside `3_Notebooks` so the relative paths to `2_Datasets` and `4_Deployment` resolve.
Section 2.3 writes `4_Deployment/researchly_model.joblib`, which the web app loads.

## Running the web app

```bash
cd 4_Deployment
./run.sh          # Windows: run.bat
```

See `4_Deployment/README.md` for Docker and Streamlit Community Cloud options.
