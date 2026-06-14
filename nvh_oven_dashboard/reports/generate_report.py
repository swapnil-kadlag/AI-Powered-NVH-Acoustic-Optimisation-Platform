"""
reports/generate_report.py — Phase 9: PDF Report Generator
===========================================================
Renders a multi-page NVH analysis PDF using Jinja2 HTML template
and WeasyPrint for PDF conversion.

Usage:
  python reports/generate_report.py
  → outputs: reports/NVH_Report_MHC_<date>.pdf
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML, CSS

BASE  = Path(__file__).resolve().parent.parent
SIM   = BASE / "simulation"
DATA  = BASE / "data"
COMP  = BASE / "competition"
RPT   = BASE / "reports"
TMPL  = RPT / "templates"


# ── helpers ────────────────────────────────────────────────────
def _csv(path: Path) -> list[dict]:
    return json.loads(pd.read_csv(path).to_json(orient="records"))


def _json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Source mechanism descriptions ──────────────────────────────
SRC_MECHANISMS = {
    "convection_fan":  "Aerodynamic blade-pass tone + unbalance → structural",
    "cooling_fan":     "Axial fan BPF tonal + grille airborne radiation",
    "door_seal":       "Airborne leakage through door perimeter gap",
    "magnetron":       "100 Hz magnetic hum → panel radiation",
    "transformer":     "100/200 Hz electromagnetic hum → base plate",
}


def build_context() -> dict:
    """Assemble all data into a Jinja2 template context dict."""

    # ── Source ranking ──────────────────────────────────────────
    src_df = pd.read_csv(SIM / "source_ranking.csv")
    sources = json.loads(
        src_df[src_df["source"] != "total"]
        .sort_values("energy_share_pct", ascending=False)
        .to_json(orient="records")
    )
    max_share = max(s["energy_share_pct"] for s in sources) if sources else 1

    # ── Paths ───────────────────────────────────────────────────
    paths = _csv(SIM / "path_contributions.csv")
    spr   = _csv(SIM / "spr_matrix.csv")

    # ── Mitigation / reduction estimate ────────────────────────
    est  = _json(SIM / "reduction_estimate.json")
    mit  = _csv(SIM / "mitigation_matrix.csv")

    # Re-build selected_fixes list with full detail for template
    BASELINE = 60.0
    selected, spend, level = [], 0.0, BASELINE
    deduped = (pd.read_csv(SIM / "mitigation_matrix.csv")
               .drop_duplicates("engineering_fix")
               .sort_values("roi_dba_per_usd", ascending=False))
    budget = 35.0
    for _, row in deduped.iterrows():
        if spend + row["cost_usd"] <= budget:
            selected.append({
                "fix":        row["engineering_fix"],
                "dba":        float(row["dba_reduction"]),
                "cost":       float(row["cost_usd"]),
                "difficulty": row["difficulty"],
                "lead_weeks": int(row["lead_weeks"]),
                "roi":        float(row["roi_dba_per_usd"]),
            })
            spend += row["cost_usd"]
            level -= row["dba_reduction"] * 0.80
    predicted = round(max(level, 40.0), 1)

    # ── Optimisation ────────────────────────────────────────────
    surrogate_metrics = _csv(SIM / "surrogate_metrics.csv")
    shap_imp  = _csv(SIM / "shap_importance.csv")
    max_shap  = max(s["mean_abs_shap"] for s in shap_imp) if shap_imp else 1
    sweet     = _json(SIM / "sweet_spot.json")

    # ── Sound pack ──────────────────────────────────────────────
    bom       = _csv(SIM / "soundpack_bom.csv")
    sp_summary = _json(SIM / "soundpack_summary.json")

    # ── Competitors ─────────────────────────────────────────────
    try:
        competitors = _csv(COMP / "competitor_data.csv")
    except Exception:
        competitors = []

    return {
        "report_date":      datetime.now().strftime("%d %B %Y"),
        "predicted_dba":    predicted,
        "total_reduction":  round(BASELINE - predicted, 1),
        "budget_used":      round(spend, 2),
        "target_met":       predicted <= 52.0,
        "num_fixes":        len(selected),
        "selected_fixes":   selected,
        "sources":          sources,
        "max_share":        max_share,
        "src_mechanisms":   SRC_MECHANISMS,
        "paths":            paths,
        "spr":              [r for r in spr if r.get("source") != "total"],
        "surrogate_metrics": surrogate_metrics,
        "shap_importance":  shap_imp,
        "max_shap":         max_shap,
        "sweet_spot":       sweet,
        "bom":              bom,
        "sp_summary":       sp_summary,
        "competitors":      competitors,
    }


def render_html(context: dict) -> str:
    """Render the Jinja2 HTML template with context data."""
    env  = Environment(loader=FileSystemLoader(str(TMPL)))
    tmpl = env.get_template("nvh_report.html")
    return tmpl.render(**context)


def export_pdf(html_content: str, output_path: Path) -> None:
    """Convert rendered HTML to PDF using WeasyPrint."""
    HTML(string=html_content, base_url=str(TMPL)).write_pdf(str(output_path))


def run_report_generator() -> Path:
    print("=" * 62)
    print("  MHC Oven — PDF Report Generator  (Phase 9)")
    print("=" * 62)

    print("\n[1] Building report context …")
    ctx = build_context()
    print(f"    Sources: {len(ctx['sources'])} | Paths: {len(ctx['paths'])} | "
          f"Fixes: {ctx['num_fixes']} | Predicted: {ctx['predicted_dba']} dBA")

    print("\n[2] Rendering HTML …")
    html = render_html(ctx)
    print(f"    HTML size: {len(html):,} chars")

    # Save HTML for debugging
    html_path = RPT / "nvh_report_debug.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"    Debug HTML → {html_path}")

    print("\n[3] Converting to PDF (WeasyPrint) …")
    date_str  = datetime.now().strftime("%Y%m%d")
    pdf_path  = RPT / f"NVH_Report_MHC_{date_str}.pdf"
    export_pdf(html, pdf_path)

    size_kb = pdf_path.stat().st_size / 1024
    print(f"    PDF → {pdf_path}  ({size_kb:.1f} KB)")
    print("\n✅  Phase-9 report generation complete")
    return pdf_path


if __name__ == "__main__":
    out = run_report_generator()
