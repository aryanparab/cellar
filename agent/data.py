"""
data.py — loads the wine dataset and exposes SQL-style query tools.

Strategy: load CSV/Excel once into a pandas DataFrame at startup.
All filtering uses pandas (exact match, range, contains) — NOT vector search.
This gives deterministic, accurate results for structured wine data.

Content-based similarity:
  build_similarity_index() concatenates text columns into a 'content' string,
  applies TF-IDF, appends normalised numeric features, then pre-computes a
  cosine similarity matrix (n_wines × n_wines) stored in _sim_matrix.
  get_similar_wines(wine_id, top_k) returns the top-k nearest neighbours.
"""
import re
import io
import json
import hashlib
import numpy as np

import requests
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from typing import Optional

# ── Singletons ────────────────────────────────────────────────────────────────
_df: Optional[pd.DataFrame] = None
_sim_matrix: Optional[np.ndarray] = None   # shape (n, n), float32
_id_to_pos:  dict[int, int]       = {}     # wine id  →  positional row in _df

# ── Cache paths ───────────────────────────────────────────────────────────────
_CACHE_DIR         = Path(__file__).parent.parent / "cache"
_SIM_MATRIX_PATH   = _CACHE_DIR / "sim_matrix.npy"
_SIM_META_PATH     = _CACHE_DIR / "sim_meta.json"

# Columns used to build the content string (text features)
_TEXT_COLS    = ["name", "producer", "varietal", "appellation", "region", "country", "color"]
# Columns used as numeric features (weighted lower than text)
_NUMERIC_COLS = ["abv", "retail", "volume_ml", "vintage"]

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1Bkv3Jb_8YuLUG2rWUhJhQBdaGjQCMFfwF9oJ5jrYDSA"
    "/export?format=csv&gid=0"
)


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load wine data from local file or Google Sheets CSV export."""
    global _df
    if path:
        if path.endswith(".xlsx") or path.endswith(".xls"):
            _df = pd.read_excel(path)
        else:
            _df = pd.read_csv(path)
    else:
        resp = requests.get(SHEET_URL, timeout=15)
        resp.raise_for_status()
        _df = pd.read_csv(io.StringIO(resp.text))

    # Normalise column names: lowercase, strip spaces, replace spaces with _
    _df.columns = [c.strip().lower().replace(" ", "_") for c in _df.columns]
    _clean_numeric_columns()
    # Reset to a clean 0-based positional index so iloc == matrix row
    _df.reset_index(drop=True, inplace=True)
    print(f"📦 Loaded {len(_df)} wines | columns: {list(_df.columns)}")
    return _df


def _clean_numeric_columns():
    """Coerce price/rating/points columns to numeric, stripping $ signs etc."""
    global _df
    for col in _df.columns:
        if any(k in col for k in ("price", "rating", "point", "score", "year", "vintage")):
            # Convert to string, strip non-numeric (except decimal), and coerce
            _df[col] = (
                _df[col]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
            )
            _df[col] = pd.to_numeric(_df[col], errors="coerce")


def get_schema() -> dict:
    """Return column names and sample values — used by the agent as a tool."""
    if _df is None:
        return {}
    schema = {}
    for col in _df.columns:
        sample = _df[col].dropna().head(3).tolist()
        schema[col] = {
            "dtype": str(_df[col].dtype),
            "sample_values": [str(s) for s in sample],
            "null_count": int(_df[col].isna().sum()),
        }
    return schema


def query_wines(
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    region_contains: Optional[str] = None,
    country_contains: Optional[str] = None,
    variety_contains: Optional[str] = None,
    name_contains: Optional[str] = None,
    sort_by: Optional[str] = None,       # column name
    sort_desc: bool = True,
    limit: int = 20,
) -> list[dict]:
    """
    SQL-style filter over the wine DataFrame.
    Returns a list of dicts (rows) matching all supplied filters.
    Only the most relevant columns are returned to keep LLM context small.
    """
    if _df is None:
        return []

    mask = pd.Series([True] * len(_df), index=_df.index)

    # --- price filters ---
    price_col = _find_col(["price"])
    if price_col:
        if max_price is not None:
            mask &= _df[price_col].fillna(9999) <= max_price
        if min_price is not None:
            mask &= _df[price_col].fillna(0) >= min_price

    # --- rating / points filter ---
    rating_col = _find_col(["rating", "point", "score"])
    if rating_col and min_rating is not None:
        mask &= _df[rating_col].fillna(0) >= min_rating

    # --- text filters (case-insensitive contains) ---
    region_col = _find_col(["region", "appellation", "area", "origin"])
    if region_col and region_contains:
        mask &= _df[region_col].astype(str).str.contains(region_contains, case=False, na=False)

    country_col = _find_col(["country"])
    if country_col and country_contains:
        mask &= _df[country_col].astype(str).str.contains(country_contains, case=False, na=False)

    variety_col = _find_col(["variety", "varietal", "grape", "type"])
    if variety_col and variety_contains:
        mask &= _df[variety_col].astype(str).str.contains(variety_contains, case=False, na=False)

    name_col = _find_col(["name", "wine", "title", "cuvee", "label"])
    if name_col and name_contains:
        mask &= _df[name_col].astype(str).str.contains(name_contains, case=False, na=False)

    result = _df[mask].copy()

    # --- sort ---
    if sort_by and sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=not sort_desc, na_position="last")
    elif rating_col and rating_col in result.columns:
        result = result.sort_values(rating_col, ascending=False, na_position="last")

    result = result.head(limit)

    # Return only non-empty columns so the LLM context stays compact
    result = result.dropna(axis=1, how="all")
    return result.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")


def _find_col(keywords: list[str]) -> Optional[str]:
    """Find the first DataFrame column whose name contains any of the keywords."""
    if _df is None:
        return None
    for kw in keywords:
        for col in _df.columns:
            if kw in col:
                return col
    return None


def _dataset_hash() -> str:
    """MD5 of the DataFrame contents — changes whenever any cell changes."""
    row_hashes = pd.util.hash_pandas_object(_df, index=False)
    return hashlib.md5(row_hashes.values.tobytes()).hexdigest()


def _load_cache(expected_hash: str) -> bool:
    """
    Try to restore _sim_matrix and _id_to_pos from disk.
    Returns True only if the cache exists AND was built from the same dataset.
    """
    global _sim_matrix, _id_to_pos
    if not _SIM_MATRIX_PATH.exists() or not _SIM_META_PATH.exists():
        return False
    try:
        meta = json.loads(_SIM_META_PATH.read_text())
        if meta.get("hash") != expected_hash:
            print("📦 Dataset changed — similarity cache invalidated, recomputing…")
            return False
        _sim_matrix = np.load(str(_SIM_MATRIX_PATH))
        _id_to_pos  = {int(k): int(v) for k, v in meta["id_to_pos"].items()}
        print(
            f"⚡ Similarity index loaded from cache | "
            f"{_sim_matrix.shape[0]} × {_sim_matrix.shape[0]} wines"
        )
        return True
    except Exception as exc:
        print(f"⚠️  Cache load failed ({exc}) — recomputing…")
        return False


def _save_cache(dataset_hash: str) -> None:
    """Persist _sim_matrix (.npy) and _id_to_pos + hash (JSON) to disk."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(_SIM_MATRIX_PATH), _sim_matrix)
        meta = {"hash": dataset_hash, "id_to_pos": _id_to_pos}
        _SIM_META_PATH.write_text(json.dumps(meta))
        size_mb = _SIM_MATRIX_PATH.stat().st_size / 1_048_576
        print(f"💾 Similarity index cached to {_CACHE_DIR} ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"⚠️  Could not save similarity cache: {exc}")


def build_similarity_index() -> None:
    """
    Pre-compute a cosine similarity matrix over all wines, with disk caching.

    On startup:
      - Hash the loaded DataFrame.
      - If cache/sim_matrix.npy + cache/sim_meta.json exist and the hash
        matches, load from disk (milliseconds) and return early.
      - Otherwise run the full pipeline, then write the cache for next time.

    Pipeline (when cache is cold or stale):
      1. Concatenate _TEXT_COLS into a single 'content' string per wine.
      2. TF-IDF vectorise (unigrams + bigrams, sublinear TF scaling).
      3. MinMax-normalise _NUMERIC_COLS and append as weighted sparse columns.
      4. Compute cosine similarity → dense (n × n) float32 matrix.
      5. Build _id_to_pos mapping: wine 'id' value → positional row index.
      6. Save matrix + metadata to disk.
    """
    global _sim_matrix, _id_to_pos
    if _df is None:
        print("⚠️  build_similarity_index called before load_dataset — skipping")
        return

    # ── Cache check ───────────────────────────────────────────────────────────
    h = _dataset_hash()
    if _load_cache(h):
        return   # warm cache hit — done in <10 ms

    # ── 1. Build content string ───────────────────────────────────────────────
    present_text = [c for c in _TEXT_COLS if c in _df.columns]
    if present_text:
        content = (
            _df[present_text]
            .fillna("")
            .astype(str)
            .apply(lambda row: " ".join(row.values), axis=1)
        )
    else:
        content = pd.Series([""] * len(_df))

    # ── 2. TF-IDF on text ────────────────────────────────────────────────────
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,      # log(1+tf) dampens very common terms
        strip_accents="unicode",
    )
    text_matrix = tfidf.fit_transform(content)   # sparse (n, vocab)

    # ── 3. Normalised numeric features ───────────────────────────────────────
    present_num = [c for c in _NUMERIC_COLS if c in _df.columns]
    if present_num:
        num_data = (
            _df[present_num]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        scaler     = MinMaxScaler()
        num_scaled = scaler.fit_transform(num_data).astype(np.float32)
        # Weight numerics at ~20 % of the text contribution
        num_sparse = sp.csr_matrix(num_scaled * 0.2)
        combined   = sp.hstack([text_matrix, num_sparse], format="csr")
    else:
        combined = text_matrix

    # ── 4. Cosine similarity ──────────────────────────────────────────────────
    _sim_matrix = sk_cosine_similarity(combined).astype(np.float32)

    # ── 5. id → positional row mapping ───────────────────────────────────────
    if "id" in _df.columns:
        _id_to_pos = {
            int(v): pos
            for pos, v in enumerate(_df["id"].values)
            if pd.notna(v)
        }
    else:
        _id_to_pos = {pos: pos for pos in range(len(_df))}

    print(
        f"✅ Similarity index built | "
        f"{_sim_matrix.shape[0]} wines × {_sim_matrix.shape[0]} wines | "
        f"text cols: {present_text} | numeric cols: {present_num}"
    )

    # ── 6. Persist for next startup ───────────────────────────────────────────
    _save_cache(h)


def get_similar_wines(wine_id: int, top_k: int = 6) -> list[dict]:
    """
    Return the top_k most content-similar wines to wine_id.

    Looks up the wine's row in the pre-computed cosine similarity matrix,
    sorts all other wines by descending similarity, and returns the top_k
    as a list of dicts (same shape as query_wines output).
    """
    if _sim_matrix is None or _df is None:
        return []
    if wine_id not in _id_to_pos:
        return []

    pos    = _id_to_pos[wine_id]
    scores = _sim_matrix[pos]                        # shape (n,)

    # Sort descending; exclude the wine itself (score == 1.0 at pos)
    ranked = np.argsort(scores)[::-1]
    ranked = [i for i in ranked if i != pos][:top_k]

    result = _df.iloc[ranked].copy()
    result = result.dropna(axis=1, how="all")
    return result.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")


def get_wine_id_by_name(name: str) -> Optional[int]:
    """
    Resolve a wine name to its catalog ID for similarity lookups.
    Tries exact match first (case-insensitive), then falls back to
    the closest partial match so the LLM can pass a wine name naturally.
    """
    if _df is None:
        return None
    name_col = _find_col(["name", "wine", "title"])
    if not name_col:
        return None

    col = _df[name_col].astype(str).str.strip()

    # 1. Exact match (case-insensitive)
    exact = col.str.lower() == name.strip().lower()
    if exact.any():
        row = _df[exact].iloc[0]
    else:
        # 2. Partial match — pick first hit
        partial = col.str.contains(name.strip(), case=False, na=False, regex=False)
        if not partial.any():
            return None
        row = _df[partial].iloc[0]

    if "id" in _df.columns and pd.notna(row.get("id")):
        return int(row["id"])
    # Fallback: use positional index
    return int(_df[exact if exact.any() else partial].index[0])


def get_dataframe() -> Optional[pd.DataFrame]:
    return _df
