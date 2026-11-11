"""Text denoising shared by the notebook, the training script and the web app.

This is the same `denoise()` used in the notebook (Section 1.1). It lives in its own
module so the saved model (researchly_model.joblib) can be loaded by any program:
a pickled scikit-learn pipeline stores the cleaner by *reference* (module + name),
so the module must be importable wherever the model is loaded.
"""
import re
import unicodedata

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()

# Standard English stop-words plus words that appear in almost every abstract
DOMAIN_STOP = {"paper", "study", "article", "results", "result", "show", "shows",
               "use", "using", "used", "propose", "proposed", "based", "approach"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP

_STEM_CACHE = {}


def stem(word: str) -> str:
    if word not in _STEM_CACHE:
        _STEM_CACHE[word] = _stemmer.stem(word)
    return _STEM_CACHE[word]


def denoise(text: str) -> str:
    """Return a cleaned, lower-cased, stop-word-free, stemmed version of `text`."""

    # 1. Unicode normalisation: curly quotes/ligatures -> plain forms; fancy hyphens -> '-'
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text)
    # 2. Boiler-plate label at the very start ("Abstract", "Summary")
    text = re.sub(r"^\s*(?:abstract|summary)\s*[:.\-]?\s*", " ", text, flags=re.I)
    # 3. URLs, DOIs, e-mail addresses
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+\.\S+|doi:\s*\S+", " ", text, flags=re.I)
    # 4. LaTeX / inline maths such as $f(R)$ and $\sqrt{T}$
    text = re.sub(r"\$[^$]*\$", " ", text)
    # 5. Publisher copyright notices ("Copyright (c) 2008 John Wiley & Sons, Ltd.")
    text = re.sub(r"(?:copyright\s*)?©[^.]*?(?:ltd|b\.v|inc|elsevier|wiley|springer)[^.]*\.?",
                  " ", text, flags=re.I)
    text = re.sub(r"all rights reserved\.?", " ", text, flags=re.I)
    # 6. Lower-case
    text = text.lower()
    # 7. Keep letters only (drops digits, punctuation, symbols); hyphens split words
    text = re.sub(r"[^a-z\s]", " ", text)
    # 8. Tokenise, drop short tokens + stop-words, stem
    tokens = [stem(t) for t in text.split() if len(t) > 2 and t not in STOP_WORDS]
    return " ".join(tokens)
