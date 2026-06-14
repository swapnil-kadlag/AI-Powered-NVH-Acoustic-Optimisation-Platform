# AI-Powered NVH Optimisation Platform

**9 dBA noise reduction** on a domestic cooking appliance · IEC 60704-1 compliance · Python 3.12

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Live Demo

Deploy to Streamlit Cloud in 3 steps — see [Deployment](#deployment) below.

## Tech Stack

| Layer | Tools |
|---|---|
| Simulation | ESI VA One (FEM/SEA aerovibro-acoustics) |
| DOE | 2^(4-1) Res-IV FFD + Centre Points + LHS (261 runs) |
| ML / Optimisation | XGBoost · scikit-learn · SHAP · pymoo (NSGA-II) · SALib |
| Agents & RAG | LangGraph · BM25 |
| Backend | FastAPI · Uvicorn |
| Frontend | Streamlit · Plotly |
| Reporting | Jinja2 · WeasyPrint |
| Containers | Docker · docker-compose |

## Project Structure

```
nvh_oven_dashboard/
├── app.py                    ← Streamlit Cloud entry point
├── dashboard/app.py          ← Full 7-tab Streamlit dashboard
├── api/main.py               ← FastAPI REST backend (15 endpoints)
├── agents/nvh_agents.py      ← LangGraph multi-agent system
├── models/surrogate_model.py ← XGBoost + RF + NSGA-II
├── simulation/               ← SPR/TPA engine + outputs
├── data/                     ← DOE dataset (261 runs) + sensitivity
├── rag_engine/               ← BM25 RAG + knowledge base
├── reports/                  ← PDF report generator
├── requirements.txt          ← Python dependencies
├── packages.txt              ← System dependencies (WeasyPrint)
├── .streamlit/config.toml   ← Dark theme config
├── Dockerfile                ← Multi-stage production build
└── docker-compose.yml        ← API + dashboard services
```

## Deployment

### Option A — Streamlit Cloud (Recommended, Free)

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Fill in:
   - **Repository**: `your-username/nvh-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy** — live in ~3 minutes

### Option B — Docker (Local / On-Premise)

```bash
docker compose up --build
# Dashboard → http://localhost:8501
# API       → http://localhost:8000
# API Docs  → http://localhost:8000/docs
```

### Option C — Local Python

```bash
pip install -r requirements.txt
# Terminal 1 — API
uvicorn api.main:app --port 8000 --reload
# Terminal 2 — Dashboard
streamlit run dashboard/app.py
```

## Key Results

| Metric | Value |
|---|---|
| Baseline noise | 60.0 dBA |
| Target | ≤ 52.0 dBA |
| Predicted (after 5 fixes) | **50.8 dBA** |
| Reduction | **−9.2 dBA** |
| NSGA-II Pareto front | 200 solutions, 44–46 dBA |
| XGBoost R² (dBA) | 0.869 |
| Dominant factor (Sobol ST) | Blower RPM (ST = 0.647) |
