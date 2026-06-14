"""
noise_decomposition.py — Phase 2: NVH Source-Path-Receiver Engine
==================================================================
Performs full SPR (Source-Path-Receiver) noise decomposition for
the MHC Oven using synthetic measurement data from Phase 1.

Physics basis:
  - Logarithmic power addition:  L_total = 10·log10(Σ 10^(Li/10))
  - TPA path contribution:       L_path  = L_source + NTF_dB + coupling_factor
  - Insertion loss:              L_treated = L_untreated - IL(f)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT  = BASE / "simulation"

SOURCES = ["convection_fan", "cooling_fan", "door_seal",
           "magnetron", "transformer"]

FREQ_BANDS = np.array([
    20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,
    800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000
])

# ── helpers ────────────────────────────────────────────────────
def log_add(levels):
    """10·log10(Σ 10^(Li/10))  — energy summation of SPL values."""
    return 10.0 * np.log10(np.sum(10.0 ** (np.array(levels) / 10.0)))

# ══════════════════════════════════════════════════════════════
# A — Load & normalise Phase-1 data
# ══════════════════════════════════════════════════════════════
def load_data():
    spectra = pd.read_parquet(DATA / "source_spectra.parquet")   # index=freq_Hz, cols=sources
    summary_raw = pd.read_csv(DATA / "source_dba_summary.csv", index_col=0)
    # Normalise summary into long form
    summary = summary_raw.reset_index().rename(
        columns={"index": "source", "overall_dBA": "overall_dba",
                 "Unnamed: 0": "source"})
    if "source" not in summary.columns:
        summary = summary_raw.reset_index()
        summary.columns = ["source", "overall_dba", "contribution_pct"]
    summary["mode"] = "combined"
    ntf = pd.read_csv(DATA / "ntf_data.csv")
    return spectra, summary, ntf

# ══════════════════════════════════════════════════════════════
# B — Source ranking
# ══════════════════════════════════════════════════════════════
def compute_source_ranking(summary):
    df = summary.sort_values("overall_dba", ascending=False).reset_index(drop=True)
    total = log_add(df["overall_dba"].values)
    df["energy_share_pct"] = (10.0**(df["overall_dba"]/10.0) /
                               10.0**(total/10.0) * 100.0)
    df["rank"] = range(1, len(df)+1)
    df["total_dba"] = total
    return df

def build_waterfall(spectra, ranked):
    """Return dict {source: Series(freq→spl)}."""
    wf = {}
    for src in ranked["source"]:
        if src in spectra.columns:
            wf[src] = spectra[src]
    return wf

# ══════════════════════════════════════════════════════════════
# C — Path definitions
# ══════════════════════════════════════════════════════════════
STRUCTURAL_PATHS = [
    {"pid":"S1","name":"Fan Mount → Top Panel",    "type":"structural",
     "source":"convection_fan","ntf_col":"NTF_fan_bracket_dBperN",      "cf":-2.5,
     "desc":"Fan unbalance → bracket → top panel bending radiation"},
    {"pid":"S2","name":"Fan Mount → Side Panels",  "type":"structural",
     "source":"convection_fan","ntf_col":"NTF_fan_bracket_dBperN",      "cf":-4.0,
     "desc":"Fan vibration → side-wall flexural radiation"},
    {"pid":"S3","name":"Magnetron → Rear Wall",    "type":"structural",
     "source":"magnetron","ntf_col":"NTF_magnetron_chassis_dBperN",     "cf":-3.0,
     "desc":"Magnetron 100 Hz magnetic force → rear panel"},
    {"pid":"S4","name":"Transformer → Base Plate", "type":"structural",
     "source":"transformer","ntf_col":"NTF_transformer_pad_dBperN",     "cf":-1.5,
     "desc":"Transformer 100/200 Hz hum → base → floor radiation"},
    {"pid":"S5","name":"Cooling Fan → Rear Grille","type":"structural",
     "source":"cooling_fan","ntf_col":"NTF_rear_panel_center_dBperN",   "cf":-3.5,
     "desc":"Cooling fan BPF through rear grille"},
]

AIRBORNE_PATHS = [
    {"pid":"A1","name":"Door Seal Gap → Front",    "type":"airborne",
     "source":"door_seal","ntf_col":"NTF_door_latch_dBperN",            "cf": 0.0,
     "desc":"Direct airborne radiation through door perimeter gap"},
    {"pid":"A2","name":"Duct Leakage → Sides",     "type":"airborne",
     "source":"convection_fan","ntf_col":"NTF_fan_bracket_dBperN",      "cf":-6.0,
     "desc":"Fan airborne noise through duct-to-cavity leak"},
    {"pid":"A3","name":"Vent Grille → Rear",       "type":"airborne",
     "source":"cooling_fan","ntf_col":"NTF_rear_panel_center_dBperN",   "cf":-2.0,
     "desc":"Cooling fan direct airborne radiation through rear grille"},
    {"pid":"A4","name":"Magnetron Cavity → Interior","type":"airborne",
     "source":"magnetron","ntf_col":"NTF_magnetron_chassis_dBperN",     "cf":-8.0,
     "desc":"Magnetron EMI-induced acoustic radiation inside cavity"},
]

# ══════════════════════════════════════════════════════════════
# D — TPA path contributions
# ══════════════════════════════════════════════════════════════
def compute_path_contributions(ranked, ntf):
    """
    Scalar TPA: L_path = L_source_dBA + mean_NTF + coupling_factor
    NTF columns are in dB/N; mean taken over all freq bands.
    """
    src_lookup = dict(zip(ranked["source"], ranked["overall_dba"]))
    records = []
    for path in STRUCTURAL_PATHS + AIRBORNE_PATHS:
        src_dba = src_lookup.get(path["source"], 45.0)
        if path["ntf_col"] in ntf.columns:
            ntf_mean = ntf[path["ntf_col"]].mean()
        else:
            ntf_mean = -15.0
        path_dba = src_dba + ntf_mean + path["cf"]
        records.append({
            "path_id":   path["pid"],
            "path_name": path["name"],
            "path_type": path["type"],
            "source":    path["source"],
            "ntf_col":   path["ntf_col"],
            "coupling_db": path["cf"],
            "source_dba":  round(src_dba, 1),
            "ntf_mean_db": round(ntf_mean, 1),
            "path_dba":    round(path_dba, 1),
            "description": path["desc"],
        })
    df = pd.DataFrame(records).sort_values("path_dba", ascending=False).reset_index(drop=True)
    df["path_rank"] = df.index + 1
    return df

# ══════════════════════════════════════════════════════════════
# E — SPR matrix
# ══════════════════════════════════════════════════════════════
def build_spr_matrix(path_df):
    records = {}
    for src in SOURCES:
        row = {}
        for pt in ["structural","airborne"]:
            sub = path_df[(path_df["source"]==src)&(path_df["path_type"]==pt)]
            row[pt] = round(log_add(sub["path_dba"].values),1) if not sub.empty else np.nan
        vals = [v for v in row.values() if not np.isnan(v)]
        row["total"] = round(log_add(vals),1) if vals else np.nan
        records[src] = row
    spr = pd.DataFrame(records).T.reset_index()
    spr.columns = ["source","structural_dba","airborne_dba","total_dba"]
    spr = spr.sort_values("total_dba", ascending=False).reset_index(drop=True)
    spr["rank"] = spr.index+1
    return spr

def top_pairs(path_df, n=3):
    top = path_df.head(n).copy()
    top["priority"] = ["🔴 Critical","🟠 High","🟡 Medium"][:n]
    return top

# ══════════════════════════════════════════════════════════════
# F — Mitigation database & matrix
# ══════════════════════════════════════════════════════════════
MIT_DB = {
    ("convection_fan","structural"):[
        {"fix":"Aerodynamic 44-blade skewed redesign",          "dba":3.5,"usd":18,"diff":"Medium","wks":12,"mech":"Reduced BPF tonal energy"},
        {"fix":"Anti-vib mount (8 Shore-A rubber grommets)",   "dba":2.0,"usd":4, "diff":"Low",   "wks":4, "mech":"Break structural path S1/S2"},
        {"fix":"Fan speed -10% + impeller re-match",           "dba":2.5,"usd":2, "diff":"Low",   "wks":2, "mech":"Fan noise ∝ RPM⁵; -10% ≈ -2 dBA"},
    ],
    ("convection_fan","airborne"):[
        {"fix":"Duct sealing + flexible boot joint",           "dba":1.5,"usd":3, "diff":"Low",   "wks":3, "mech":"Eliminate A2 duct leakage"},
        {"fix":"Melamine liner inside fan duct (25 mm)",       "dba":2.0,"usd":6, "diff":"Medium","wks":6, "mech":"Attenuate airborne BPF harmonics"},
    ],
    ("magnetron","structural"):[
        {"fix":"Add shield mass 0.3 kg steel plate",           "dba":4.0,"usd":8, "diff":"Low",   "wks":2, "mech":"Mass law: +6 dB/octave shielding"},
        {"fix":"Magnetron constrained-layer damping ring",     "dba":2.5,"usd":5, "diff":"Medium","wks":8, "mech":"Structural loss factor η≈0.15"},
    ],
    ("cooling_fan","structural"):[
        {"fix":"EC blower (eliminates axial BPF tone)",        "dba":3.0,"usd":12,"diff":"High",  "wks":20,"mech":"Centrifugal vs axial: tone removal"},
        {"fix":"Rear grille foam-backed perforated treatment", "dba":1.5,"usd":3, "diff":"Low",   "wks":4, "mech":"Grille IL improvement 500-2k Hz"},
    ],
    ("cooling_fan","airborne"):[
        {"fix":"Rear grille reactive splitter muffler",        "dba":2.0,"usd":5, "diff":"Medium","wks":8, "mech":"Dissipative muffler at outlet"},
    ],
    ("transformer","structural"):[
        {"fix":"Silicone isolators K=2000 N/m (fn≈5 Hz)",     "dba":2.5,"usd":6, "diff":"Low",   "wks":3, "mech":"Isolation ratio r=20 → IL≈26 dB@100Hz"},
        {"fix":"Encapsulated damped housing",                  "dba":1.5,"usd":4, "diff":"Low",   "wks":4, "mech":"Increased radiation resistance"},
    ],
    ("door_seal","airborne"):[
        {"fix":"Dual-lip EPDM gasket upgrade",                 "dba":1.5,"usd":2, "diff":"Low",   "wks":4, "mech":"Reduce airborne transmission at perimeter"},
        {"fix":"Door latch pre-load increase",                 "dba":0.8,"usd":1, "diff":"Low",   "wks":2, "mech":"Better seal contact pressure"},
    ],
}

def build_mitigation_matrix(top):
    records=[]
    for _,row in top.iterrows():
        key=(row["source"],row["path_type"])
        for i,fix in enumerate(MIT_DB.get(key,[])):
            records.append({
                "priority":    row["priority"],
                "source":      row["source"],
                "path":        row["path_name"],
                "path_type":   row["path_type"],
                "engineering_fix": fix["fix"],
                "dba_reduction":   fix["dba"],
                "cost_usd":        fix["usd"],
                "difficulty":      fix["diff"],
                "lead_weeks":      fix["wks"],
                "mechanism":       fix["mech"],
                "roi_dba_per_usd": round(fix["dba"]/max(fix["usd"],0.01),3),
            })
    df=pd.DataFrame(records)
    if not df.empty:
        df=df.sort_values("dba_reduction",ascending=False).reset_index(drop=True)
    return df

def estimate_reduction(mit_df, budget=35.0):
    BASELINE=60.0
    selected,spend,level=[],0.0,BASELINE
    for _,r in mit_df.drop_duplicates("engineering_fix")\
                      .sort_values("roi_dba_per_usd",ascending=False).iterrows():
        if spend+r["cost_usd"]<=budget:
            selected.append(r["engineering_fix"])
            spend+=r["cost_usd"]
            level-=r["dba_reduction"]*0.80
    return {"baseline_dba":BASELINE,"predicted_dba":round(max(level,40),1),
            "total_reduction":round(BASELINE-max(level,40),1),"target_dba":52.0,
            "target_met":max(level,40)<=52.0,"budget_used_usd":round(spend,2),
            "num_fixes":len(selected),"selected_fixes":selected}

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run_decomposition():
    print("="*62)
    print("  MHC Oven — NVH Noise Decomposition  (Phase 2)")
    print("="*62)

    print("\n[A] Loading data …")
    spectra, summary, ntf = load_data()
    print(f"    {len(summary)} sources | {len(ntf)} NTF freq points")

    print("\n[B] Source ranking …")
    ranked = compute_source_ranking(summary)
    print(ranked[["rank","source","overall_dba","energy_share_pct"]].to_string(index=False))

    print("\n[C] Waterfall …")
    wf = build_waterfall(spectra, ranked)
    print(f"    {len(wf)} source spectra loaded")

    print("\n[D] TPA path contributions …")
    path_df = compute_path_contributions(ranked, ntf)
    print(path_df[["path_rank","path_name","source","path_dba"]].to_string(index=False))

    print("\n[E] SPR matrix …")
    spr = build_spr_matrix(path_df)
    print(spr.to_string(index=False))

    print("\n[F] Top 3 critical pairs …")
    tp = top_pairs(path_df, 3)
    print(tp[["priority","path_name","source","path_dba"]].to_string(index=False))

    print("\n[G] Mitigation matrix …")
    mit = build_mitigation_matrix(tp)
    print(mit[["engineering_fix","dba_reduction","cost_usd","difficulty"]].head(8).to_string(index=False))

    print("\n[H] Budget-optimised reduction estimate …")
    est = estimate_reduction(mit, 35.0)
    print(f"    {est['baseline_dba']} → {est['predicted_dba']} dBA  "
          f"(−{est['total_reduction']} dBA)  "
          f"Target {'✅' if est['target_met'] else '❌'}  "
          f"Spend ${est['budget_used_usd']}")

    OUT.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUT/"source_ranking.csv",index=False)
    path_df.to_csv(OUT/"path_contributions.csv",index=False)
    spr.to_csv(OUT/"spr_matrix.csv",index=False)
    tp.to_csv(OUT/"top_source_path_pairs.csv",index=False)
    mit.to_csv(OUT/"mitigation_matrix.csv",index=False)
    with open(OUT/"reduction_estimate.json","w") as f:
        json.dump(est, f, indent=2)
    print(f"\n✅  Phase-2 outputs → {OUT}/")

    return {"sources_ranked":ranked,"waterfall":wf,"path_df":path_df,
            "spr_matrix":spr,"top_pairs":tp,"mitigation_matrix":mit,
            "reduction_estimate":est}

if __name__=="__main__":
    results = run_decomposition()
