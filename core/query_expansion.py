""""
query_expansion.py
-------------------
Robust query expansion module (FIXED + CLEAN NLTK setup)
"""

import logging
from typing import Optional

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import nltk

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")

ensure_nltk()
# ─────────────────────────────────────────────
# SAFE NLTK SETUP (FIXED PROPERLY)
# ─────────────────────────────────────────────

def safe_nltk_download():
    required = ["wordnet", "stopwords", "punkt", "averaged_perceptron_tagger"]

    for pkg in required:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
                logger.info(f"[NLTK] Downloaded {pkg}")
            except Exception as e:
                logger.warning(f"[NLTK] Failed downloading {pkg}: {e}")


safe_nltk_download()

STOP_WORDS = set(stopwords.words("english"))


# ─────────────────────────────────────────────
# 1. WORDNET EXPANSION
# ─────────────────────────────────────────────

def expand_with_wordnet(query: str, max_synonyms_per_word: int = 2) -> str:
    tokens = word_tokenize(query.lower())
    expanded = list(tokens)

    for token in tokens:
        if token in STOP_WORDS or not token.isalpha():
            continue

        syns = set()

        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")

                if name.lower() != token and name.lower() not in STOP_WORDS:
                    syns.add(name.lower())

                if len(syns) >= max_synonyms_per_word:
                    break
            if len(syns) >= max_synonyms_per_word:
                break

        expanded.extend(list(syns))

    # remove duplicates (preserve order)
    seen, out = set(), []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return " ".join(out)


# ─────────────────────────────────────────────
# 2. WORD2VEC EXPANDER
# ─────────────────────────────────────────────

class Word2VecExpander:
    def __init__(self, topn: int = 3):
        self.topn = topn
        self.wv = None

        try:
            import gensim.downloader as api
            logger.info("[Word2Vec] Loading glove-wiki-gigaword-100...")
            self.wv = api.load("glove-wiki-gigaword-100")
        except Exception as e:
            logger.warning(f"[Word2Vec] Disabled: {e}")

    def expand(self, query: str) -> str:
        if self.wv is None:
            return query

        tokens = word_tokenize(query.lower())
        extra = []

        for t in tokens:
            if t in STOP_WORDS or not t.isalpha():
                continue

            if t in self.wv:
                sims = self.wv.most_similar(t, topn=self.topn)
                extra.extend([w for w, _ in sims])

        final = tokens + extra

        seen, out = set(), []
        for t in final:
            if t not in seen:
                seen.add(t)
                out.append(t)

        return " ".join(out)


# ─────────────────────────────────────────────
# 3. PRF EXPANDER
# ─────────────────────────────────────────────

class PRFExpander:
    def __init__(self, text_encoder, faiss_index, id_to_text, top_k=5, max_terms=5):
        self.encoder = text_encoder
        self.index = faiss_index
        self.id_to_text = id_to_text
        self.top_k = top_k
        self.max_terms = max_terms

    def expand(self, query: str) -> str:
        vec = self.encoder.encode_query(query).astype("float32").reshape(1, -1)
        _, ids = self.index.search(vec, self.top_k)

        text = " ".join(
            self.id_to_text.get(int(i), "")
            for i in ids[0]
            if i != -1
        )

        tokens = word_tokenize(text.lower())
        q_tokens = set(word_tokenize(query.lower()))

        freq = {}
        for t in tokens:
            if t.isalpha() and t not in STOP_WORDS and t not in q_tokens:
                freq[t] = freq.get(t, 0) + 1

        top = sorted(freq, key=freq.get, reverse=True)[: self.max_terms]

        return query + " " + " ".join(top)


# ─────────────────────────────────────────────
# UNIFIED EXPANDER
# ─────────────────────────────────────────────

class QueryExpander:
    def __init__(self, strategy="wordnet",
                 text_encoder=None, faiss_index=None, id_to_text=None):

        self.strategy = strategy

        self.w2v = Word2VecExpander() if strategy in ["word2vec", "combined"] else None

        self.prf = None
        if strategy in ["prf", "combined"] and text_encoder and faiss_index and id_to_text:
            self.prf = PRFExpander(text_encoder, faiss_index, id_to_text)

    def expand(self, query: str) -> str:

        if self.strategy == "wordnet":
            return expand_with_wordnet(query)

        if self.strategy == "word2vec":
            return self.w2v.expand(query) if self.w2v else query

        if self.strategy == "prf":
            return self.prf.expand(query) if self.prf else query

        # combined
        out = expand_with_wordnet(query)
        if self.w2v:
            out = self.w2v.expand(out)
        if self.prf:
            out = self.prf.expand(out)

        return out


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    qe = QueryExpander("wordnet")
    print(qe.expand("dog running fast"))