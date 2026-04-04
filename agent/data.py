"""
data.py — loads the wine dataset and exposes SQL-style query tools.

Strategy: load CSV/Excel once into a pandas DataFrame at startup.
All filtering uses pandas (exact match, range, contains) — NOT vector search.
This gives deterministic, accurate results for structured wine data.
"""
import re
import io
import numpy as np

import requests
import pandas as pd
from typing import Optional

# Singleton dataframe
_df: Optional[pd.DataFrame] = None

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


def get_dataframe() -> Optional[pd.DataFrame]:
    return _df
