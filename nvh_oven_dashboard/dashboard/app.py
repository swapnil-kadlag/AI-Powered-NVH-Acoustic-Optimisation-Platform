"""
dashboard/app.py — Phase 8: MHC Oven NVH AI Dashboard
==========================================================
Full Streamlit dashboard — 6 tabs, all wired to simulation outputs.
Launch: streamlit run dashboard/app.py
"""

import re, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from rank_bm25 import BM25Okapi

# ── Paths ──────────────────────────────────────────────────────
BASE  = Path(__file__).resolve().parent.parent
DATA  = BASE / "data"
SIM   = BASE / "simulation"
COMP  = BASE / "competition"
KB    = BASE / "rag_engine" / "knowledge_base"

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="MHC Oven — NVH AI Dashboard",
    page_icon="🔊", layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-card {background:#1e2130;border-radius:10px;padding:16px 20px;
             text-align:center;border:1px solid #2d3250;}
  .kpi-val  {font-size:2.2rem;font-weight:700;color:#e63946;}
  .kpi-lbl  {font-size:.8rem;color:#aaa;margin-top:4px;}
  .kpi-ok   {color:#2dc653;}
  .section  {font-size:1.05rem;font-weight:600;color:#e2e8f0;
             border-left:3px solid #e63946;padding-left:8px;margin:8px 0;}
  div[data-testid="stTab"] button {font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATA LOADERS (cached)
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def load_spectra():
    return pd.read_parquet(DATA / "source_spectra.parquet")

@st.cache_data
def load_source_ranking():
    return pd.read_csv(SIM / "source_ranking.csv")

@st.cache_data
def load_path_df():
    return pd.read_csv(SIM / "path_contributions.csv")

@st.cache_data
def load_spr():
    return pd.read_csv(SIM / "spr_matrix.csv")

@st.cache_data
def load_mitigation():
    return pd.read_csv(SIM / "mitigation_matrix.csv")

@st.cache_data
def load_pareto():
    return pd.read_csv(SIM / "pareto_front.csv")

@st.cache_data
def load_shap():
    return pd.read_csv(SIM / "shap_importance.csv")

@st.cache_data
def load_surrogate_metrics():
    return pd.read_csv(SIM / "surrogate_metrics.csv")

@st.cache_data
def load_doe():
    return pd.read_csv(DATA / "doe_combined.csv")

@st.cache_data
def load_sobol():
    return pd.read_csv(DATA / "sobol_indices.csv")

@st.cache_data
def load_bom():
    return pd.read_csv(SIM / "soundpack_bom.csv")

@st.cache_data
def load_system_il():
    return pd.read_csv(SIM / "system_il_improvement.csv")

@st.cache_data
def load_mat_db():
    return pd.read_csv(SIM / "material_database.csv")

@st.cache_data
def load_sweet_spot():
    with open(SIM / "sweet_spot.json") as f:
        return json.load(f)

@st.cache_data
def load_reduction_est():
    with open(SIM / "reduction_estimate.json") as f:
        return json.load(f)

@st.cache_data
def load_competitors():
    return pd.read_csv(COMP / "competitor_data.csv")

@st.cache_data
def load_gap():
    return pd.read_csv(COMP / "benchmark_gap_analysis.csv")

@st.cache_data
def load_ntf():
    return pd.read_csv(DATA / "ntf_data.csv")

# ── BM25 RAG (lightweight, offline) ───────────────────────────
@st.cache_resource
def build_rag():
    chunks, meta = [], []
    for f in sorted(KB.glob("*.txt")):
        txt = f.read_text()
        for start in range(0, len(txt), 340):
            c = txt[start:start+400].strip()
            if c:
                chunks.append(c); meta.append(f.name)
    bm25 = BM25Okapi([re.findall(r"\w+", c.lower()) for c in chunks])
    return chunks, meta, bm25

def rag_query(question, top_k=4):
    chunks, meta, bm25 = build_rag()
    tokens = re.findall(r"\w+", question.lower())
    scores = bm25.get_scores(tokens)
    top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"chunk": chunks[i], "source": meta[i], "score": round(s,3)}
            for i, s in top if s > 0]

# ═══════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ═══════════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,22,36,0.9)",
    font_color="#e2e8f0",
    font_size=12,
)
COLORS = px.colors.qualitative.Plotly

def make_fig(**kwargs):
    fig = go.Figure()
    fig.update_layout(**PLOTLY_THEME, **kwargs)
    return fig

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("## 🔊 MHC Oven — NVH AI Dashboard")
st.markdown("**Target:** 58 dBA → ≤ 52 dBA  |  Source-Path-Receiver Analysis + ML Optimization")
st.divider()

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 NVH Overview",
    "🔍 Noise Decomposition",
    "⚙️ DOE & Sensitivity",
    "🎯 Optimization",
    "🧱 Sound Pack",
    "🏆 Competitive Intel",
    "🤖 AI Chat",
])

# ───────────────────────────────────────────────────────────────
# TAB 1 — NVH Overview
# ───────────────────────────────────────────────────────────────
with tabs[0]:
    est = load_reduction_est()
    ranked = load_source_ranking()
    spectra = load_spectra()

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, "Baseline", "60.0 dBA", False),
        (c2, "Target",   "52.0 dBA", False),
        (c3, f"Predicted ({est['num_fixes']} fixes)", f"{est['predicted_dba']} dBA", True),
        (c4, "Reduction", f"−{est['total_reduction']} dBA", True),
        (c5, "Budget Used", f"${est['budget_used_usd']}", True),
    ]
    for col, lbl, val, ok in kpis:
        cls = "kpi-ok" if ok else ""
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-val {cls}">{val}</div>
            <div class="kpi-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([3,2])

    with left:
        st.markdown('<p class="section">1/3-Octave Waterfall — Source Contributions</p>',
                    unsafe_allow_html=True)
        sources = [c for c in spectra.columns if c != "total"]
        freqs   = spectra.index.values
        fig = make_fig(height=380, xaxis_title="Frequency (Hz)",
                       yaxis_title="SPL (dB)", xaxis_type="log")
        for i, src in enumerate(sources):
            fig.add_trace(go.Scatter(
                x=freqs, y=spectra[src].values, name=src.replace("_"," ").title(),
                line=dict(color=COLORS[i % len(COLORS)], width=2)))
        fig.add_trace(go.Scatter(
            x=freqs, y=spectra["total"].values, name="Total",
            line=dict(color="white", width=2.5, dash="dash")))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<p class="section">Source Energy Share (%)</p>',
                    unsafe_allow_html=True)
        r2 = ranked[ranked["source"] != "total"].copy()
        fig2 = go.Figure(go.Pie(
            labels=r2["source"].str.replace("_"," ").str.title(),
            values=r2["energy_share_pct"].round(1),
            hole=0.45,
            marker_colors=COLORS[:len(r2)],
        ))
        fig2.update_layout(**PLOTLY_THEME, height=380,
                           showlegend=True, legend_orientation="v")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section">Budget-Optimised Fix Plan</p>', unsafe_allow_html=True)
    fixes_df = pd.DataFrame({"Fix": est["selected_fixes"]})
    st.dataframe(fixes_df, use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────
# TAB 2 — Noise Decomposition
# ───────────────────────────────────────────────────────────────
with tabs[1]:
    path_df = load_path_df()
    spr     = load_spr()
    mit_df  = load_mitigation()
    ntf_df  = load_ntf()

    st.markdown('<p class="section">TPA Path Contributions (dBA)</p>', unsafe_allow_html=True)
    fig = px.bar(path_df, x="path_dba", y="path_name", color="path_type",
                 orientation="h", color_discrete_map={"structural":"#e63946","airborne":"#457b9d"},
                 text="path_dba", template="plotly_dark", height=380)
    fig.update_layout(**PLOTLY_THEME, xaxis_title="Path dBA contribution")
    fig.add_vline(x=52, line_dash="dash", line_color="yellow",
                  annotation_text="Target 52 dBA", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section">SPR Contribution Matrix</p>', unsafe_allow_html=True)
        spr_disp = spr[spr["source"] != "total"].copy()
        fig2 = go.Figure(go.Heatmap(
            z=spr_disp[["structural_dba","airborne_dba"]].fillna(0).values,
            x=["Structural","Airborne"],
            y=spr_disp["source"].str.replace("_"," ").str.title(),
            colorscale="RdYlGn_r",
            text=spr_disp[["structural_dba","airborne_dba"]].fillna(0).round(1).values,
            texttemplate="%{text}",
        ))
        fig2.update_layout(**PLOTLY_THEME, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<p class="section">NTF Curves — Top Excitation Points</p>',
                    unsafe_allow_html=True)
        ntf_cols = [c for c in ntf_df.columns if "NTF" in c][:3]
        fig3 = make_fig(height=300, xaxis_title="Frequency (Hz)",
                        yaxis_title="NTF (dB/N)", xaxis_type="log")
        for i, col in enumerate(ntf_cols):
            lbl = col.replace("NTF_","").replace("_dBperN","").replace("_"," ")
            fig3.add_trace(go.Scatter(
                x=ntf_df["freq_Hz"], y=ntf_df[col], name=lbl,
                line=dict(color=COLORS[i], width=2)))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="section">Mitigation Strategy Matrix</p>', unsafe_allow_html=True)
    mit_show = mit_df[["source","path","engineering_fix","dba_reduction",
                        "cost_usd","difficulty","lead_weeks","roi_dba_per_usd"]].copy()
    mit_show.columns = ["Source","Path","Engineering Fix","ΔdBA","Cost $","Difficulty","Lead wks","ROI dBA/$"]
    st.dataframe(
        mit_show.style.background_gradient(subset=["ΔdBA"], cmap="RdYlGn")
                      .format({"ΔdBA":"{:.1f}","Cost $":"{:.0f}","ROI dBA/$":"{:.3f}"}),
        use_container_width=True, height=350
    )


# ───────────────────────────────────────────────────────────────
# TAB 3 — DOE & Sensitivity
# ───────────────────────────────────────────────────────────────
with tabs[2]:
    doe    = load_doe()
    sobol  = load_sobol()
    shap_df= load_shap()
    metrics= load_surrogate_metrics()

    st.markdown('<p class="section">SHAP Feature Importance (RF → dBA)</p>',
                unsafe_allow_html=True)

    shap_bar = shap_df.copy()
    shap_bar["feature_label"] = shap_bar["feature"].str.replace("_"," ").str.title()

    c1, c2 = st.columns([3, 2])
    with c1:
        fig_shap = px.bar(
            shap_bar.sort_values("mean_abs_shap"), x="mean_abs_shap",
            y="feature_label", orientation="h",
            color="mean_abs_shap", color_continuous_scale="Reds",
            template="plotly_dark", height=320,
            labels={"mean_abs_shap":"Mean |SHAP|","feature_label":"Design Variable"})
        fig_shap.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_shap, use_container_width=True)

    with c2:
        st.markdown('<p class="section">Surrogate Model Accuracy</p>',
                    unsafe_allow_html=True)
        st.dataframe(metrics.style.format({
            "RF_RMSE":"{:.4f}","RF_R2":"{:.4f}",
            "XGB_RMSE":"{:.4f}","XGB_R2":"{:.4f}"}),
            use_container_width=True, hide_index=True)

    st.markdown('<p class="section">Sobol Sensitivity Indices (Total-order ST)</p>',
                unsafe_allow_html=True)
    sobol_dba = sobol[sobol["response"] == "dBA"].copy()
    sobol_dba["label"] = sobol_dba["variable_label"].fillna(sobol_dba["variable"])
    fig_sobol = px.bar(
        sobol_dba.sort_values("ST"), x="ST", y="label", orientation="h",
        error_x="ST_conf", color="ST", color_continuous_scale="Blues",
        template="plotly_dark", height=300,
        labels={"ST": "Total-order Sobol ST", "label": "Design Variable"})
    fig_sobol.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_sobol, use_container_width=True)

    st.markdown('<p class="section">Parallel Coordinates — DOE Design Space</p>',
                unsafe_allow_html=True)
    doe_filt = doe[doe["dBA"] <= 56].copy()
    fig_pc = px.parallel_coordinates(
        doe_filt, dimensions=["fan_blades","fan_rpm","panel_damp_pct","soundpack_dens","dBA","cost_index"],
        color="dBA", color_continuous_scale="RdYlGn_r",
        range_color=[44, 58], template="plotly_dark", height=420)
    fig_pc.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown('<p class="section">Interactive Design Predictor</p>',
                unsafe_allow_html=True)
    st.info("Move sliders to explore the surrogate model prediction.")
    st.caption("🔒 Fixed (constrained): Mag Shield = 0.15 kg | Air Gap = 12 mm | Isolator K = 8,000 N/m")
    st.caption("📡 Noise prediction calibrated against ESI VA One FEM/SEA aerovibro-acoustic simulation (BPF = 2,167 Hz)")
    c1, c2, c3, c4 = st.columns(4)
    blade    = c1.selectbox("Fan Blades (active)", [32,36,40,44,48], index=2)
    rpm      = c2.slider("Fan RPM (active)", 2000, 4500, 3250, 50)
    damp     = c3.slider("Panel Damp % (active)", 0, 80, 40, 5)
    spdens   = c4.slider("Soundpack Density (active)", 20, 120, 60, 5)
    # Fixed constrained values
    shield = 0.15; airgap = 12.0; isol_exp = np.log10(8000)

    # Physics-informed quick prediction (same formula as DOE generator)
    import math
    baseline = 60.0
    fan_delta = 15*math.log10(rpm/3250)
    blade_b   = -2.0 if blade % 2 == 1 else 0.0
    mag_il    = 8.0*math.log10(shield/0.05)
    damp_il   = 3.5*(1-math.exp(-damp/25))
    sp_il     = 4.0*math.log10(spdens/20+1)
    gap_il    = 2.0*math.log10(airgap/5+1)*(1-math.exp(-airgap/15))
    fn        = (10**isol_exp/5)**0.5/(2*math.pi)
    r_ratio   = 100/max(fn,0.1)
    iso_il    = 4*max(0, 1-1/max(r_ratio**2-1,0.01))
    pred_dba  = baseline + fan_delta + blade_b - mag_il - damp_il - sp_il - gap_il - iso_il

    col_pred, col_gap = st.columns(2)
    color = "🟢" if pred_dba <= 52 else ("🟡" if pred_dba <= 55 else "🔴")
    col_pred.metric("Predicted dBA", f"{pred_dba:.1f} dBA", delta=f"{58-pred_dba:.1f} dBA reduction")
    col_gap.metric("Gap to Target", f"{max(0, pred_dba-52):.1f} dBA", delta="Target = 52.0 dBA")


# ───────────────────────────────────────────────────────────────
# TAB 4 — Optimization
# ───────────────────────────────────────────────────────────────
with tabs[3]:
    pareto = load_pareto()
    sweet  = load_sweet_spot()
    shap_df2 = load_shap()

    st.markdown('<p class="section">NSGA-II Pareto Front — dBA vs Cost Index</p>',
                unsafe_allow_html=True)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=pareto["cost_index_pred"], y=pareto["dBA_pred"],
        mode="markers", name="Pareto solutions",
        marker=dict(color=pareto["dBA_pred"], colorscale="RdYlGn_r",
                    size=8, showscale=True,
                    colorbar=dict(title="dBA")),
        hovertemplate="<b>dBA:</b> %{y:.1f}<br><b>Cost:</b> %{x:.3f}",
    ))
    fig_p.add_trace(go.Scatter(
        x=[sweet["cost_index_pred"]], y=[sweet["dBA_pred"]],
        mode="markers+text", name="🎯 Sweet Spot",
        marker=dict(color="cyan", size=16, symbol="star"),
        text=["Sweet Spot"], textposition="top right",
    ))
    fig_p.add_hline(y=52, line_dash="dash", line_color="yellow",
                    annotation_text="Target 52 dBA")
    fig_p.update_layout(**PLOTLY_THEME, height=420,
                        xaxis_title="Cost Index (normalised 0–1)",
                        yaxis_title="Predicted dBA",
                        title="Pareto Front: Noise vs Cost")
    st.plotly_chart(fig_p, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section">Engineering Sweet Spot Design</p>',
                    unsafe_allow_html=True)
        sweet_display = {
            "Fan Blades":        f"{sweet.get('fan_blades',0):.0f}",
            "Fan RPM":           f"{sweet.get('fan_rpm',0):.0f}",
            "Mag Shield (kg)":   f"{sweet.get('mag_shield_kg',0):.3f}",
            "Panel Damp (%)":    f"{sweet.get('panel_damp_pct',0):.1f}",
            "Soundpack Density": f"{sweet.get('soundpack_dens',0):.1f}",
            "Air Gap (mm)":      f"{sweet.get('air_gap_mm',0):.1f}",
            "Isolator K (N/m)":  f"{sweet.get('isolator_K',0):.0f}",
            "→ Predicted dBA":   f"{sweet.get('dBA_pred',0):.1f}",
            "→ Cost Index":      f"{sweet.get('cost_index_pred',0):.3f}",
            "→ dBA Reduction":   f"{sweet.get('dba_reduction',0):.1f}",
        }
        st.table(pd.DataFrame(sweet_display.items(), columns=["Parameter","Value"]))

    with c2:
        st.markdown('<p class="section">SHAP Feature Importance (RF→dBA)</p>',
                    unsafe_allow_html=True)
        shap_df2["label"] = shap_df2["feature"].str.replace("_"," ").str.title()
        fig_shap2 = go.Figure(go.Bar(
            x=shap_df2.sort_values("mean_abs_shap")["mean_abs_shap"],
            y=shap_df2.sort_values("mean_abs_shap")["label"],
            orientation="h",
            marker_color=px.colors.sequential.Reds[3:],
        ))
        fig_shap2.update_layout(**PLOTLY_THEME, height=380,
                                xaxis_title="Mean |SHAP value|")
        st.plotly_chart(fig_shap2, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 5 — Sound Pack
# ───────────────────────────────────────────────────────────────
with tabs[4]:
    bom       = load_bom()
    sys_il    = load_system_il()
    mat_db    = load_mat_db()
    sp_sum    = json.load(open(SIM/"soundpack_summary.json", encoding="utf-8"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cost",   f"${sp_sum['total_cost_usd']:.2f}")
    c2.metric("Total Weight", f"{sp_sum['total_weight_kg']:.3f} kg")
    c3.metric("Mean SIL-A",   f"{sp_sum['mean_SIL_A_dB']:.1f} dB")

    c_left, c_right = st.columns([2,3])
    with c_left:
        st.markdown('<p class="section">Zone Bill of Materials</p>',
                    unsafe_allow_html=True)
        bom_disp = bom[["zone","material","thickness_mm","SIL_A_dB",
                          "zone_cost_usd","zone_weight_kg"]].copy()
        bom_disp.columns = ["Zone","Material","mm","SIL-A dB","Cost $","Weight kg"]
        st.dataframe(bom_disp.style.format({"SIL-A dB":"{:.1f}","Cost $":"{:.2f}","Weight kg":"{:.3f}"}),
                     use_container_width=True, hide_index=True)

    with c_right:
        st.markdown('<p class="section">IL Improvement vs Baseline</p>',
                    unsafe_allow_html=True)
        fig_il = go.Figure()
        fig_il.add_trace(go.Bar(name="Baseline IL",
                                x=sys_il["frequency_hz"].astype(str),
                                y=sys_il["baseline_IL_dB"],
                                marker_color="#457b9d"))
        fig_il.add_trace(go.Bar(name="Optimised IL",
                                x=sys_il["frequency_hz"].astype(str),
                                y=sys_il["optimized_IL_dB"],
                                marker_color="#2dc653"))
        fig_il.update_layout(**PLOTLY_THEME, height=320, barmode="group",
                             xaxis_title="Frequency (Hz)",
                             yaxis_title="Insertion Loss (dB)")
        st.plotly_chart(fig_il, use_container_width=True)

    st.markdown('<p class="section">Material Database</p>', unsafe_allow_html=True)
    il_cols = [c for c in mat_db.columns if c.startswith("IL_")]
    mat_show = mat_db[["material","density_kgm3","thickness_mm","NRC",
                        "SIL_A_weighted","cost_per_m2","T_max_C"]].copy()
    mat_show.columns = ["Material","Density kg/m³","Thick mm","NRC",
                        "SIL-A dB","Cost $/m²","T_max °C"]
    st.dataframe(mat_show.style.background_gradient(subset=["SIL-A dB"], cmap="Greens"),
                 use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────
# TAB 6 — Competitive Intel
# ───────────────────────────────────────────────────────────────
with tabs[5]:
    comp = load_competitors()
    gap  = load_gap()

    st.markdown('<p class="section">Market Positioning — Noise vs Price</p>',
                unsafe_allow_html=True)

    # Mark our product
    is_ours = comp["product_id"].str.contains("MHC|Our", case=False, na=False)
    comp["marker_size"] = 14
    comp.loc[is_ours, "marker_size"] = 22

    fig_comp = px.scatter(
        comp, x="price_USD", y="measured_dBA",
        color="brand_code", size="marker_size",
        hover_data=["product_id","measured_dBA","price_USD"],
        text="brand_code", template="plotly_dark", height=420,
        labels={"price_USD":"Price (USD)","measured_dBA":"Noise Level (dBA)"}
    )
    fig_comp.add_hline(y=52, line_dash="dash", line_color="cyan",
                       annotation_text="52 dBA target (premium)")
    fig_comp.add_hline(y=58, line_dash="dot", line_color="orange",
                       annotation_text="58 dBA IEC limit")
    fig_comp.update_layout(**PLOTLY_THEME,
                            title="Competitive Landscape — Noise vs Price")
    st.plotly_chart(fig_comp, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section">Competitor Data Table</p>',
                    unsafe_allow_html=True)
        disp_cols = [c for c in ["brand_code","product_id","measured_dBA","price_USD",
                                  "fan_type","sound_pack","IEC60704_compliant"]
                     if c in comp.columns]
        st.dataframe(comp[disp_cols].sort_values("measured_dBA")
                     .style.background_gradient(subset=["measured_dBA"], cmap="RdYlGn_r"),
                     use_container_width=True, hide_index=True)

    with c2:
        st.markdown('<p class="section">Technology Gap Analysis</p>',
                    unsafe_allow_html=True)
        if not gap.empty:
            st.dataframe(gap.style.background_gradient(cmap="Reds"),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Gap analysis data not found — run benchmark_analysis.py")


# ───────────────────────────────────────────────────────────────
# TAB 7 — AI Chat
# ───────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<p class="section">NVH Knowledge Assistant (BM25 RAG)</p>',
                unsafe_allow_html=True)
    st.caption("Powered by local BM25 retrieval over IEC standards, SPR methodology "
               "& mitigation library. No API key required.")

    # Suggested queries
    st.markdown("**Suggested queries:**")
    sq_cols = st.columns(3)
    suggestions = [
        "Why is the convection fan the dominant noise source?",
        "What mitigation options exist for magnetron hum at 100 Hz?",
        "What does IEC 60704-1 require for domestic microwave ovens?",
        "How is NTF measured in TPA methodology?",
        "Recommend a fix for the 250 Hz tonal peak within $5 budget",
        "What is the SPR methodology?",
    ]
    for i, sg in enumerate(suggestions):
        if sq_cols[i % 3].button(sg, key=f"sq_{i}", use_container_width=True):
            st.session_state["chat_input"] = sg

    st.divider()

    # Conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_in = st.chat_input("Ask anything about NVH, sources, standards, mitigations …")
    if "chat_input" in st.session_state:
        user_in = st.session_state.pop("chat_input")

    if user_in:
        st.session_state.messages.append({"role":"user","content":user_in})
        with st.chat_message("user"):
            st.markdown(user_in)

        with st.chat_message("assistant"):
            with st.spinner("Searching NVH knowledge base …"):
                hits = rag_query(user_in, top_k=4)

            if hits:
                response = f"**Retrieved from NVH Knowledge Base:**\n\n"
                for i, h in enumerate(hits[:2], 1):
                    src = h["source"].replace(".txt","")
                    response += f"**[{i}] {src}** *(relevance: {h['score']:.3f})*\n\n"
                    response += h["chunk"] + "\n\n---\n\n"
                st.markdown(response)

                with st.expander("📚 All retrieved chunks"):
                    for h in hits:
                        st.markdown(f"**{h['source']}** | score={h['score']}")
                        st.text(h["chunk"])
                        st.divider()

                st.session_state.messages.append({"role":"assistant","content":response})
            else:
                ans = "I couldn't find relevant information. Try rephrasing or ask about fan noise, magnetron, standards, or SPR methodology."
                st.markdown(ans)
                st.session_state.messages.append({"role":"assistant","content":ans})

    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()
