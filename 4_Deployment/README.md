# Researchly – Research Papers Expert System (Deployment)

A small web app that reads a research-paper abstract and says whether the paper uses
**AI** methods or **classical** methods. It is the deployment of the design from
**Question 1** (cleaning + features + baseline) and **Question 2** (model comparison + tuned Linear SVM).

* Paste one abstract → label, signal strength, and the words that pushed the decision
* Upload a CSV/JSON of many abstracts → predictions table you can download

Model: `denoise()` cleaning → TF-IDF (words + word pairs) → Linear SVM (C = 0.5, balanced class weights).
Expected quality: ≈ 0.86 macro-F1 in repeated cross-validation.

## What is in this folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web interface |
| `preprocessing.py` | The `denoise()` text cleaner (identical to the notebook) |
| `researchly_model.joblib` | The trained model, saved by the notebook (Section 2.3) |
| `train_model.py` | Re-trains the model from `../2_Datasets/research_abstracts.json` |
| `requirements.txt` | Python packages (scikit-learn is pinned – see *Troubleshooting*) |
| `run.sh` / `run.bat` | One-command launchers for macOS/Linux and Windows |
| `Dockerfile` | Container build for any Docker host |

## Option 1 – Run it on your computer (easiest)

You need **Python 3.10 or newer**.

**Windows:** double-click `run.bat` (or run it in a terminal).
**macOS / Linux:** `./run.sh`

The launcher creates a private environment, installs the packages and opens the app at
<http://localhost:8501>. Stop it with `Ctrl + C`.

Manual version of the same thing:

```bash
cd 4_Deployment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Option 2 – Docker

```bash
cd 4_Deployment
docker build -t researchly .
docker run -p 8501:8501 researchly
```

Then open <http://localhost:8501>. The same image runs on any container host
(Azure Container Apps, AWS App Runner, Google Cloud Run, a VPS, …) – point it at port 8501.

## Option 3 – Streamlit Community Cloud (free public link)

1. Put the contents of this folder at the **root** of a GitHub repository
   (`app.py`, `preprocessing.py`, `train_model.py`, `researchly_model.joblib`, `requirements.txt`, `.streamlit/`).
2. Go to <https://share.streamlit.io>, sign in with GitHub and choose **New app**.
3. Select the repository, branch, and set **Main file path** to `app.py`, then **Deploy**.

If you leave out `researchly_model.joblib`, the app trains it on first start – this needs the
dataset, so also copy `research_abstracts.json` and change `DEFAULT_DATA` in `train_model.py` to point to it.

## Re-training the model

```bash
python train_model.py            # uses ../2_Datasets/research_abstracts.json
python train_model.py --data path/to/other_abstracts.json
```

The data file needs the columns `Abstract` and `Label` (`ai` or `classical`). Restart the app afterwards.

## Using the app

1. Open the **Single abstract** tab, paste an abstract (or pick an example) and press **Classify**.
2. Read the label and the *decision score*: 0 means undecided; the further from 0, the stronger the signal
   (< 0.3 borderline, 0.3–0.8 moderate, > 0.8 strong). It is a margin, **not a probability**.
3. The two tables list the cleaned (stemmed) terms that pushed towards AI or classical.
4. For many papers use **Batch file**: upload a CSV/JSON with an `Abstract` column and download the results.

## Troubleshooting

* **`InconsistentVersionWarning` / model fails to load** – the model file is tied to the scikit-learn version it
  was trained with (1.9.1). Either install the pinned version from `requirements.txt` or run `python train_model.py`
  to rebuild the model with your version.
* **`ModuleNotFoundError: preprocessing`** – start the app from inside the `4_Deployment` folder.
* **Port already in use** – `streamlit run app.py --server.port 8502`.
* **Weak results on non-finance abstracts** – the training data is mostly economics/finance research; the model is
  not designed for other fields.
