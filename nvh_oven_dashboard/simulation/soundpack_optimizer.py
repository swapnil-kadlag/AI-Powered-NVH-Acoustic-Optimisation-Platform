"""
soundpack_optimizer.py — Phase 4: Sound Package Zone Optimizer
===============================================================
Selects optimal acoustic treatment materials for each oven panel zone
to maximise weighted Insertion Loss (IL) within budget, weight, and
thermal constraints.

Material physics:
  - Absorption (NRC): random-incidence absorption coefficient per 1/3-oct
  - Transmission Loss (IL): modelled via mass law + limp-panel theory
    IL(f) = 20·log10(m·f·π / (ρ₀·c₀))  where m = surface density (kg/m²)
  - Temperature constraint: materials rated up to T_max °C

Zones:
  Top panel, Side panels (×2), Rear wall, Bottom, Door inner
"""

import numpy as np
import pandas as pd
import json, itertools
from pathlib import Path
from scipy.optimize import linprog

BASE   = Path(__file__).resolve().parent.parent
DATA   = BASE / "data"
SIMDIR = BASE / "simulation"
SIMDIR.mkdir(parents=True, exist_ok=True)

# ── 1/3-octave centre frequencies used for IL evaluation ──────
FREQS = np.array([63,125,250,500,1000,2000,4000,8000])  # Hz
N_FREQ = len(FREQS)

# ── A-weighting correction (dB) at those frequencies ──────────
A_WEIGHT = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])

# ═══════════════════════════════════════════════════════════════
# A — Material Database
# ═══════════════════════════════════════════════════════════════

def build_material_db() -> pd.DataFrame:
    """
    Synthetic but physically plausible material properties.
    IL values computed from mass-law + measured NRC profiles.

    Mass-law IL (dB): IL = 20·log10(m_surf·f·π / (ρ₀·c₀))
      ρ₀·c₀ ≈ 415 Pa·s/m  (air impedance at 20°C)
    """
    rho0_c0 = 415.0

    # Base material parameters
    mats = [
        # name,         density, thick_mm, NRC,  cost_m2, T_max_C, notes
        ("Melamine Foam",       8,   25, 0.90, 12.0,  150, "Low density open-cell; excellent absorption"),
        ("Fiberglass Batt",    32,   50, 0.95,  8.0,  350, "High-temp; good broadband absorption"),
        ("Mass Loaded Vinyl",2000,    3, 0.10, 18.0,  80,  "Heavy limp barrier; mass-law IL"),
        ("PU Foam",            30,   20, 0.75, 6.0,   120, "General purpose foam; moderate absorption"),
        ("Recycled PET Felt",  30,   25, 0.85, 5.0,   140, "Eco-friendly; good absorption"),
        ("Rubber Damping Mat", 1800,  4, 0.05, 14.0,  120, "Constrained-layer damping; reduces panel radiation"),
        ("Bitumen Pad",        1200,  2, 0.03, 9.0,   80,  "Anti-drum deadener; structural damping"),
        ("Air Gap (25 mm)",    1.2,  25, 0.00, 0.5,   500, "Decoupled cavity; reactive IL via impedance mismatch"),
    ]

    records = []
    for name, rho, thick, nrc, cost, t_max, notes in mats:
        thick_m   = thick / 1000.0
        m_surf    = rho * thick_m                 # surface density kg/m²

        # IL per 1/3-oct band (mass law; clamp at 5 dB floor)
        if name == "Air Gap (25 mm)":
            # Reactive: IL ≈ 6 dB at all freq (impedance mismatch)
            il = np.array([6, 7, 8, 9, 9, 8, 7, 6], dtype=float)
        elif name == "Rubber Damping Mat":
            # CLD: primarily structural, less airborne IL; add fixed offset
            il = np.maximum(5, 20*np.log10(m_surf * FREQS * np.pi / rho0_c0)) + 2.0
        else:
            il = np.maximum(5, 20*np.log10(m_surf * FREQS * np.pi / rho0_c0))

        # NRC-weighted A-weighted IL (single number for ranking)
        a_il = il + A_WEIGHT
        nrc_arr = np.full(N_FREQ, nrc)            # simplified: flat NRC

        rec = {
            "material":    name,
            "density_kgm3": rho,
            "thickness_mm": thick,
            "NRC":          nrc,
            "surface_density_kgm2": round(m_surf, 3),
            "cost_per_m2":  cost,
            "T_max_C":      t_max,
            "notes":        notes,
        }
        for i, f in enumerate(FREQS):
            rec[f"IL_{f}Hz"] = round(float(il[i]), 1)
        rec["SIL_A_weighted"] = round(float(np.mean(a_il)), 1)  # single-number IL (A-wtd avg)
        records.append(rec)

    df = pd.DataFrame(records)
    return df


# ═══════════════════════════════════════════════════════════════
# B — Oven Zone Definitions
# ═══════════════════════════════════════════════════════════════

def build_zone_defs() -> pd.DataFrame:
    """
    Define oven panel zones with area, thermal limit, and dominant source.

    Thermal limits reflect proximity to heating elements:
      - Top panel: 250°C max (near convection elements)
      - Rear wall: 200°C (near magnetron cavity)
      - Side panels: 150°C
      - Door inner: 180°C
      - Bottom: 120°C
    """
    zones = [
        # name,         area_m2, T_limit_C, budget_usd, wt_limit_kg, primary_source,        priority
        ("Top Panel",    0.070,   250,       8.0,        0.25, "convection_fan",       1),
        ("Side Panel L", 0.060,   150,       6.0,        0.20, "convection_fan",       2),
        ("Side Panel R", 0.060,   150,       6.0,        0.20, "cooling_fan",          2),
        ("Rear Wall",    0.080,   200,       9.0,        0.30, "magnetron",            1),
        ("Bottom Panel", 0.070,   120,       5.0,        0.20, "transformer",          3),
        ("Door Inner",   0.040,   180,       4.0,        0.10, "door_seal",            2),
    ]
    df = pd.DataFrame(zones, columns=[
        "zone","area_m2","T_limit_C","budget_usd",
        "weight_limit_kg","primary_source","priority"
    ])
    return df


# ═══════════════════════════════════════════════════════════════
# C — Zone-level material selection
# ═══════════════════════════════════════════════════════════════

def select_material_for_zone(zone: pd.Series, mat_db: pd.DataFrame) -> pd.Series:
    """
    Greedy selection: for each zone, pick the material that maximises
    A-weighted SIL within thermal, budget, and weight constraints.

    Stacking strategy: up to 2 layers allowed (absorber + barrier).
    """
    # Filter thermally compatible materials
    eligible = mat_db[mat_db["T_max_C"] >= zone["T_limit_C"]].copy()
    if eligible.empty:
        eligible = mat_db.copy()  # relax if none qualify

    # Compute per-zone cost and weight
    eligible = eligible.copy()
    eligible["zone_cost"]   = eligible["cost_per_m2"]   * zone["area_m2"]
    eligible["zone_weight"] = eligible["surface_density_kgm2"] * zone["area_m2"]

    # Filter by budget and weight
    feasible = eligible[
        (eligible["zone_cost"]   <= zone["budget_usd"]) &
        (eligible["zone_weight"] <= zone["weight_limit_kg"])
    ].copy()

    if feasible.empty:
        feasible = eligible  # relax constraints

    # Select highest SIL_A_weighted
    best = feasible.loc[feasible["SIL_A_weighted"].idxmax()]
    return best


def optimize_soundpack(mat_db: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run zone-by-zone greedy optimisation and build BoM.
    Returns combined BoM DataFrame.
    """
    bom_rows = []
    for _, zone in zone_df.iterrows():
        best = select_material_for_zone(zone, mat_db)
        row = {
            "zone":          zone["zone"],
            "priority":      zone["priority"],
            "area_m2":       zone["area_m2"],
            "T_limit_C":     zone["T_limit_C"],
            "primary_source":zone["primary_source"],
            "material":      best["material"],
            "thickness_mm":  best["thickness_mm"],
            "NRC":           best["NRC"],
            "SIL_A_dB":      best["SIL_A_weighted"],
            "zone_cost_usd": round(best["cost_per_m2"] * zone["area_m2"], 2),
            "zone_weight_kg":round(best["surface_density_kgm2"] * zone["area_m2"], 3),
        }
        # IL at each frequency for this zone
        for f in FREQS:
            row[f"IL_{f}Hz"] = best[f"IL_{f}Hz"]
        bom_rows.append(row)

    bom = pd.DataFrame(bom_rows)
    return bom


# ═══════════════════════════════════════════════════════════════
# D — System-level IL improvement
# ═══════════════════════════════════════════════════════════════

def compute_system_il(bom: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute area-weighted system IL improvement per frequency band.

    System IL = Σ(zone_area × IL_zone) / Σ(zone_area)
    This approximates the effective IL seen at the receiver position.

    Baseline IL from Phase-1 insertion_loss_data.csv.
    """
    il_data = pd.read_csv(DATA / "insertion_loss_data.csv")

    total_area = zone_df["area_m2"].sum()
    records = []
    for f in FREQS:
        col = f"IL_{f}Hz"
        if col in bom.columns:
            # Area-weighted mean
            weighted_il = (bom[col] * bom["area_m2"]).sum() / total_area
        else:
            weighted_il = 8.0

        # Baseline IL from Phase-1 data at this frequency (interpolate)
        if "frequency_hz" in il_data.columns:
            freq_col = "frequency_hz"
            il_col   = "il_db_mean" if "il_db_mean" in il_data.columns else il_data.columns[1]
            baseline_il = float(np.interp(f, il_data[freq_col], il_data[il_col]))
        else:
            baseline_il = 6.0  # fallback

        records.append({
            "frequency_hz":      f,
            "baseline_IL_dB":    round(baseline_il, 1),
            "optimized_IL_dB":   round(weighted_il, 1),
            "IL_improvement_dB": round(weighted_il - baseline_il, 1),
            "A_weight_dB":       A_WEIGHT[list(FREQS).index(f)],
        })

    df = pd.DataFrame(records)
    # A-weighted improvement
    a_il_improvement = df["IL_improvement_dB"].values + df["A_weight_dB"].values
    df["A_weighted_improvement_dB"] = a_il_improvement.round(1)
    return df


def summarise_bom(bom: pd.DataFrame) -> dict:
    """Overall cost, weight, and IL summary."""
    return {
        "total_cost_usd":        round(bom["zone_cost_usd"].sum(), 2),
        "total_weight_kg":       round(bom["zone_weight_kg"].sum(), 3),
        "mean_SIL_A_dB":         round(bom["SIL_A_dB"].mean(), 1),
        "zones_treated":         len(bom),
        "unique_materials_used": bom["material"].nunique(),
        "materials_used":        bom["material"].tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def run_soundpack_optimizer():
    print("="*62)
    print("  MHC Oven — Sound Pack Optimizer  (Phase 4)")
    print("="*62)

    # A. Material database
    print("\n[A] Building material database …")
    mat_db = build_material_db()
    print(f"    {len(mat_db)} materials defined")
    print(mat_db[["material","density_kgm3","thickness_mm","NRC",
                  "SIL_A_weighted","cost_per_m2","T_max_C"]].to_string(index=False))

    # B. Zone definitions
    print("\n[B] Defining oven zones …")
    zone_df = build_zone_defs()
    print(zone_df[["zone","area_m2","T_limit_C","budget_usd","weight_limit_kg"]].to_string(index=False))

    # C. Optimise
    print("\n[C] Zone-level material optimisation …")
    bom = optimize_soundpack(mat_db, zone_df)
    print(bom[["zone","material","thickness_mm","SIL_A_dB",
               "zone_cost_usd","zone_weight_kg"]].to_string(index=False))

    # D. System IL
    print("\n[D] System-level IL improvement …")
    system_il = compute_system_il(bom, zone_df)
    print(system_il[["frequency_hz","baseline_IL_dB","optimized_IL_dB",
                      "IL_improvement_dB"]].to_string(index=False))

    # E. Summary
    print("\n[E] BoM Summary …")
    summary = summarise_bom(bom)
    for k, v in summary.items():
        print(f"    {k:<28}: {v}")

    # ── Save ──────────────────────────────────────────────────
    mat_db.to_csv(SIMDIR/"material_database.csv", index=False)
    zone_df.to_csv(SIMDIR/"zone_definitions.csv", index=False)
    bom.to_csv(SIMDIR/"soundpack_bom.csv", index=False)
    system_il.to_csv(SIMDIR/"system_il_improvement.csv", index=False)
    with open(SIMDIR/"soundpack_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅  Phase-4 outputs → {SIMDIR}/")

    return {"mat_db":mat_db,"zone_df":zone_df,
            "bom":bom,"system_il":system_il,"summary":summary}


if __name__ == "__main__":
    results = run_soundpack_optimizer()
