# Wine Voice Explorer

A voice-enabled web app that answers questions about a wine dataset using Grok AI.

## Architecture

```
Browser (Web Speech API)
    ↕ voice in/out — free, built-in
FastAPI backend (Python)
    └── Wine Agent
            ├── Tool: filter_wines  (pandas SQL-style query)
            ├── Tool: get_schema    (column inspector)
            └── Grok API (xAI)     (LLM reasoning on filtered results)
                    ↑
                Wine dataset (CSV from Google Sheets, loaded at startup)
```

**Why SQL-style queries, not vector DB?**
The dataset is small and structured. Exact/range filters (price ≤ $50, region = "Burgundy") are more
accurate than semantic similarity search for this use case. The agent filters first, then only the
matching rows are sent to the LLM — preventing hallucination.

## Setup

### 1. Clone and install

```bash
cd wine-explorer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Grok API key

```bash
cp .env.example .env
# Edit .env and add your GROK_API_KEY
# Get one at https://console.x.ai
```

### 3. Run

```bash
python main.py
# or: uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

## Using a local Excel/CSV file

If you've downloaded the wine dataset locally:

```bash
# In .env:
WINE_DATA_PATH=./wines.xlsx
```

Or pass a path at startup by editing `main.py`:
```python
load_dataset(path="wines.xlsx")
```

## API endpoints

| Endpoint | Description |
|---|---|
| `POST /api/ask` | Send `{"question": "..."}`, get `{"answer": "..."}` |
| `GET /api/schema` | See dataset columns and sample values |
| `GET /api/sample` | See 5 sample wine rows |
| `GET /api/health` | Health check |

## Voice support

Voice input uses the browser's **Web Speech API** (free, no key needed).
- Works in: Chrome, Edge, Safari
- Firefox: partial support — text input is the fallback

## Deploying

For the demo, run locally. For production:
- Set `GROK_API_KEY` as an environment variable (not in .env)
- Use `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Works on Railway, Render, Fly.io out of the box
