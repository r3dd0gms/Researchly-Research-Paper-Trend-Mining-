# ITNPA – Research Papers: Trend Mining AI (Submission Package)

| Folder | Contents |
|---|---|
| `1_Documentation/` | `Reseachly_Interpretation_Report.docx` – project documentation for Questions 1–4 |
| `2_Datasets/` | `research_abstracts.json` (1,138 labelled abstracts) and `research_corpus/research_1.txt … research_5.txt` (five full articles) |
| `3_Notebooks/` | `ITNPA.ipynb` – the full analytics flow, Q1–Q4, with all outputs saved; `lda_topics.html` – the interactive pyLDAvis map from Q4 |
| `4_Deployment/` | Streamlit web app for the Q1 & Q2 design, with its own `README.md` for easy deployment |


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
