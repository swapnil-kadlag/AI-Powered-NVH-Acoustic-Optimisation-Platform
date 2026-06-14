"""
api/main.py — Phase 7: FastAPI REST Backend
============================================
Exposes all NVH analysis outputs and the multi-agent system
as a typed REST API.  All endpoints return JSON.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /                          — health check + version
  GET  /sources                   — ranked noise source summary
  GET  /paths                     — TPA path contributions
  GET  /spr                       — SPR contribution matrix
  GET  /mitigation                — mitigation strategy matrix
  GET  /pareto                    — NSGA-II Pareto front
  GET  /sweet-spot                — engineering sweet-spot design
  GET  /soundpack                 — sound pack BoM + system IL
  GET  /competitors               — competitive benchmark data
  GET  /reduction-estimate        — budget-optimised reduction estimate
  POST /agent/query               — LangGraph multi-agent query
  POST /rag/query                 — BM25 RAG knowledge retrieval
  GET  /doe                       — DOE dataset (paginated)
  GET  /sensitivity               — Sobol + SHAP sensitivity indices
  GET  /surrogate-metrics         — RF vs XGBoost model accuracy
"""

import re, json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Project paths ──────────────────────────────────────────────
BASE  = Path(__file__).resolve().parent.parent
DATA  = BASE / "data"
SIM   = BASE / "simulation"
COMP  = BASE / "competition"
KB    = BASE / "rag_engine" / "knowledge_base"

# ── Lazy imports for heavy modules ────────────────────────────
_agent_system = None
_rag_chunks   = None
_rag_meta     = None
_rag_bm25     = None


def _get_agents():
    global _agent_system
    if _agent_system is None:
        from agents.nvh_agents import NVHAgentSystem
        _agent_system = NVHAgentSystem()
    return _agent_system


def _get_rag():
    global _rag_chunks, _rag_meta, _rag_bm25
    if _rag_bm25 is None:
        from rank_bm25 import BM25Okapi
        chunks, meta = [], []
        for f in sorted(KB.glob("*.txt")):
            txt = f.read_text()
            for start in range(0, len(txt), 340):
                c = txt[start:start+400].strip()
                if c:
                    chunks.append(c); meta.append(f.name)
        _rag_chunks = chunks
        _rag_meta   = meta
        _rag_bm25   = BM25Okapi([re.findall(r"\w+", c.lower()) for c in chunks])
    return _rag_chunks, _rag_meta, _rag_bm25


# ── Helpers ────────────────────────────────────────────────────
def _csv(fname: str) -> pd.DataFrame:
    p = SIM / fname
    if not p.exists():
        p = DATA / fname
    if not p.exists():
        p = COMP / fname
    if not p.exists():
        raise HTTPException(404, f"{fname} not generated yet — run pipeline first")
    return pd.read_csv(p)


def _json_file(fname: str) -> dict:
    p = SIM / fname
    if not p.exists():
        raise HTTPException(404, f"{fname} not generated yet")
    with open(p) as f:
        return json.load(f)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Safely serialise DataFrame (handle NaN, numpy types)."""
    return json.loads(df.to_json(orient="records"))


# ══════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="MHC Oven — NVH AI API",
    version="1.0.0",
    description="REST API for the NVH AI Dashboard — Source-Path-Receiver analysis, "
                "ML optimization, multi-agent system, and knowledge RAG.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════

class AgentQueryRequest(BaseModel):
    query: str
    return_full_state: bool = False


class RAGQueryRequest(BaseModel):
    query:  str
    top_k:  int = 4


class DesignPoint(BaseModel):
    fan_blades:      float = 40   # active
    fan_rpm:         float = 3250  # active
    mag_shield_kg:   float = 0.15  # FIXED — space constraint
    panel_damp_pct:  float = 40   # active
    soundpack_dens:  float = 60   # active
    air_gap_mm:      float = 12.0  # FIXED — tooling freeze
    isolator_K:      float = 8000  # FIXED — standard rubber foot


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["meta"])
def health():
    """Health check and version."""
    return {
        "status":    "ok",
        "product":   "MHC Oven NVH AI",
        "version":   "1.0.0",
        "baseline":  "60.0 dBA",
        "target":    "52.0 dBA",
        "endpoints": [
            "/sources", "/paths", "/spr", "/mitigation",
            "/pareto", "/sweet-spot", "/soundpack",
            "/competitors", "/reduction-estimate",
            "/doe", "/sensitivity", "/surrogate-metrics",
            "/agent/query", "/rag/query", "/predict",
        ]
    }


# ── Data endpoints ─────────────────────────────────────────────

@app.get("/sources", tags=["data"])
def get_sources():
    """Ranked noise source dBA summary with energy share."""
    df = _csv("source_ranking.csv")
    df = df[df["source"] != "total"] if "source" in df.columns else df
    return {"sources": _df_to_records(df), "count": len(df)}


@app.get("/paths", tags=["data"])
def get_paths(path_type: Optional[str] = None):
    """TPA path contributions. Filter by path_type=structural|airborne."""
    df = _csv("path_contributions.csv")
    if path_type:
        df = df[df["path_type"] == path_type]
    return {"paths": _df_to_records(df), "count": len(df)}


@app.get("/spr", tags=["data"])
def get_spr():
    """Source-Path-Receiver contribution matrix."""
    df = _csv("spr_matrix.csv")
    return {"spr_matrix": _df_to_records(df)}


@app.get("/mitigation", tags=["data"])
def get_mitigation(
    max_cost: Optional[float] = None,
    difficulty: Optional[str] = None,
    sort_by: str = Query("roi_dba_per_usd", enum=["roi_dba_per_usd","dba_reduction","cost_usd","lead_weeks"])
):
    """Mitigation strategy matrix with optional filtering."""
    df = _csv("mitigation_matrix.csv")
    if max_cost is not None:
        df = df[df["cost_usd"] <= max_cost]
    if difficulty:
        df = df[df["difficulty"].str.lower() == difficulty.lower()]
    df = df.sort_values(sort_by, ascending=(sort_by != "roi_dba_per_usd"))
    return {"mitigations": _df_to_records(df), "count": len(df)}


@app.get("/pareto", tags=["optimization"])
def get_pareto(
    max_dba: Optional[float] = None,
    max_cost: Optional[float] = None
):
    """NSGA-II Pareto front. Filter by max_dba and/or max_cost."""
    df = _csv("pareto_front.csv")
    if max_dba is not None:
        df = df[df["dBA_pred"] <= max_dba]
    if max_cost is not None:
        df = df[df["cost_index_pred"] <= max_cost]
    return {"pareto_front": _df_to_records(df), "count": len(df)}


@app.get("/sweet-spot", tags=["optimization"])
def get_sweet_spot():
    """Engineering sweet-spot design point from NSGA-II."""
    return _json_file("sweet_spot.json")


@app.get("/soundpack", tags=["data"])
def get_soundpack():
    """Sound package BoM, system IL improvement, and material database."""
    bom   = _df_to_records(_csv("soundpack_bom.csv"))
    il    = _df_to_records(_csv("system_il_improvement.csv"))
    mats  = _df_to_records(_csv("material_database.csv"))
    summ  = _json_file("soundpack_summary.json")
    return {"bom": bom, "system_il": il, "materials": mats, "summary": summ}


@app.get("/competitors", tags=["data"])
def get_competitors():
    """Competitive benchmark data — 6 market competitors."""
    try:
        df = _csv("competitor_data.csv")
    except HTTPException:
        raise HTTPException(404, "Run competition/competitor_scraper.py first")
    return {"competitors": _df_to_records(df), "count": len(df)}


@app.get("/reduction-estimate", tags=["data"])
def get_reduction_estimate():
    """Budget-optimised noise reduction estimate."""
    return _json_file("reduction_estimate.json")


@app.get("/doe", tags=["data"])
def get_doe(
    limit: int = Query(50, le=500),
    offset: int = 0,
    doe_type: Optional[str] = None
):
    """DOE dataset (paginated). doe_type=fractional_factorial_2k-1|lhs."""
    df = _csv("doe_combined.csv")
    if doe_type:
        df = df[df["doe_type"] == doe_type]
    total = len(df)
    page  = df.iloc[offset:offset+limit]
    return {"data": _df_to_records(page), "total": total,
            "limit": limit, "offset": offset}


@app.get("/sensitivity", tags=["data"])
def get_sensitivity():
    """Sobol total-order indices + SHAP feature importance."""
    sobol = _df_to_records(_csv("sobol_indices.csv"))
    shap  = _df_to_records(_csv("shap_importance.csv"))
    return {"sobol": sobol, "shap": shap}


@app.get("/surrogate-metrics", tags=["data"])
def get_surrogate_metrics():
    """RF vs XGBoost hold-out accuracy (RMSE and R²)."""
    df = _csv("surrogate_metrics.csv")
    return {"metrics": _df_to_records(df)}


# ── Prediction endpoint ────────────────────────────────────────

@app.post("/predict", tags=["optimization"])
def predict_dba(design: DesignPoint):
    """
    Physics-informed dBA prediction for a given design point.
    Uses the same parametric model as the DOE generator (Phase 1).
    """
    import math
    b = design
    baseline  = 60.0
    fan_delta = 15 * math.log10(b.fan_rpm / 3250)
    blade_b   = -2.0 if int(b.fan_blades) % 2 == 1 else 0.0
    mag_il    = 8.0  * math.log10(b.mag_shield_kg / 0.05)
    damp_il   = 3.5  * (1 - math.exp(-b.panel_damp_pct / 25))
    sp_il     = 4.0  * math.log10(b.soundpack_dens / 20 + 1)
    gap_il    = 2.0  * math.log10(b.air_gap_mm / 5 + 1) * (1 - math.exp(-b.air_gap_mm / 15))
    fn        = (b.isolator_K / 5) ** 0.5 / (2 * math.pi)
    r_ratio   = 100 / max(fn, 0.1)
    iso_il    = 4 * max(0, 1 - 1 / max(r_ratio**2 - 1, 0.01))
    pred      = baseline + fan_delta + blade_b - mag_il - damp_il - sp_il - gap_il - iso_il

    return {
        "design":        design.dict(),
        "predicted_dba": round(pred, 2),
        "reduction_dba": round(baseline - pred, 2),
        "target_met":    pred <= 52.0,
        "components": {
            "fan_aerodynamic_delta": round(fan_delta, 2),
            "blade_bonus":           round(blade_b, 2),
            "magnetron_shield_IL":   round(mag_il, 2),
            "panel_damping_IL":      round(damp_il, 2),
            "soundpack_IL":          round(sp_il, 2),
            "air_gap_IL":            round(gap_il, 2),
            "isolator_IL":           round(iso_il, 2),
        }
    }


# ── Agent endpoint ─────────────────────────────────────────────

@app.post("/agent/query", tags=["agents"])
def agent_query(req: AgentQueryRequest):
    """
    Run the LangGraph multi-agent system for an NVH query.
    Returns the final report (and optionally the full state).
    """
    system = _get_agents()
    state  = system.run(req.query)

    result = {
        "query":       req.query,
        "intent":      state.get("intent",""),
        "agent_trace": state.get("agent_trace", []),
        "report":      state.get("final_report",""),
    }
    if req.return_full_state:
        # Serialise state (exclude non-serialisable parts)
        result["source_analysis"]  = state.get("source_analysis", {})
        result["path_analysis"]    = state.get("path_analysis", {})
        result["mitigation_plan"]  = state.get("mitigation_plan", {})

    return result


# ── RAG endpoint ───────────────────────────────────────────────

@app.post("/rag/query", tags=["rag"])
def rag_query(req: RAGQueryRequest):
    """
    BM25 hybrid RAG retrieval over the NVH knowledge base.
    Returns top-k relevant chunks with source attribution.
    """
    chunks, meta, bm25 = _get_rag()
    tokens = re.findall(r"\w+", req.query.lower())
    scores = bm25.get_scores(tokens)
    top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:req.top_k]
    results = [
        {"rank": i+1, "score": round(s, 4),
         "source": meta[idx], "chunk": chunks[idx]}
        for i, (idx, s) in enumerate(top) if s > 0
    ]
    return {
        "query":   req.query,
        "results": results,
        "count":   len(results),
    }
