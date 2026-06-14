"""
Competitive Benchmarking Module — NVH AI Dashboard
====================================================
Extracts and synthesizes competitor NVH data for microwave/convection ovens.

Data sources:
1. Web-scraped publicly available specification sheets (noise ratings)
2. Synthetic data derived from published IEC/ISO test reports
3. Product reviews with measurable dBA claims (Amazon, manufacturer sites)

Competitors covered (anonymized as per industry practice):
  BrandA: Premium European brand (e.g., Miele/Siemens class)
  BrandB: Mid-range Asian OEM (e.g., Samsung/LG class)
  BrandC: Budget North American brand (e.g., Toshiba/Panasonic class)
  BrandD: Built-in specialist (e.g., Bosch class)
  OurUnit: MHC Oven (target product)

NVH metrics extracted:
  - Claimed dBA (from spec sheet)
  - Measured dBA (from test reports / reviews)
  - Dominant noise source (inferred from frequency analysis)
  - Sound pack features (from teardown/review data)
  - Price tier
  - Compliance status (IEC 60704-1)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

np.random.seed(2024)
OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC COMPETITOR DATABASE (based on real market data ranges)
# ─────────────────────────────────────────────────────────────────────────────

COMPETITOR_DATA = {
    "OurUnit_MHC": {
        "brand_code": "MHC",
        "tier": "Mid-premium",
        "price_USD": 380,
        "volume_L": 50,
        "power_W": 1200,

        # NVH Performance
        "claimed_dBA": 58.0,   # What we advertise (none currently)
        "measured_dBA": 58.0,  # Actual measured (our baseline)
        "target_dBA": 52.0,    # Project target

        # 1/3-oct band levels (A-weighted, key bands)
        "band_100Hz_dBA": 42.1,
        "band_125Hz_dBA": 43.5,
        "band_200Hz_dBA": 47.8,
        "band_250Hz_dBA": 48.2,
        "band_315Hz_dBA": 49.1,  # BPF region
        "band_500Hz_dBA": 46.3,
        "band_1kHz_dBA":  44.7,
        "band_2kHz_dBA":  42.1,
        "band_4kHz_dBA":  38.5,

        # Source breakdown (dBA)
        "dBA_convection_fan": 52.1,
        "dBA_magnetron": 47.3,
        "dBA_cooling_fan": 44.8,
        "dBA_transformer": 43.2,
        "dBA_door_seal": 39.5,

        # Design features
        "fan_blade_count": 9,
        "fan_type": "centrifugal",
        "has_magnetron_shield": False,
        "shield_mass_kg": 0.0,
        "panel_damping_pct": 0,
        "sound_pack": "none",
        "air_gap_mm": 8,
        "isolator_type": "rubber_pad",
        "isolator_stiffness_Npm": 50000,

        # Compliance
        "IEC60704_limit_dBA": 58.0,
        "IEC60704_compliant": True,
        "energy_class": "A+",

        # Market data
        "launch_year": 2022,
        "market_share_pct": 4.2,
        "customer_noise_complaints_pct": 18.5,
        "return_rate_pct": 3.1,
    },

    "BrandA_Premium": {
        "brand_code": "BrandA",
        "tier": "Premium",
        "price_USD": 850,
        "volume_L": 45,
        "power_W": 1000,

        "claimed_dBA": 44.0,  # Advertised (premium brands invest heavily)
        "measured_dBA": 46.2,  # Independently measured (slight inflation)
        "target_dBA": None,

        "band_100Hz_dBA": 34.1,
        "band_125Hz_dBA": 35.5,
        "band_200Hz_dBA": 38.8,
        "band_250Hz_dBA": 39.2,
        "band_315Hz_dBA": 40.1,
        "band_500Hz_dBA": 38.3,
        "band_1kHz_dBA":  36.7,
        "band_2kHz_dBA":  34.1,
        "band_4kHz_dBA":  30.5,

        "dBA_convection_fan": 43.1,
        "dBA_magnetron": 38.3,
        "dBA_cooling_fan": 36.8,
        "dBA_transformer": 33.2,
        "dBA_door_seal": 30.5,

        "fan_blade_count": 11,
        "fan_type": "EC_motor_centrifugal",
        "has_magnetron_shield": True,
        "shield_mass_kg": 0.35,
        "panel_damping_pct": 75,
        "sound_pack": "melamine_foam_25mm",
        "air_gap_mm": 25,
        "isolator_type": "viscoelastic",
        "isolator_stiffness_Npm": 3000,

        "IEC60704_limit_dBA": 58.0,
        "IEC60704_compliant": True,
        "energy_class": "A+++",

        "launch_year": 2023,
        "market_share_pct": 12.8,
        "customer_noise_complaints_pct": 2.1,
        "return_rate_pct": 0.8,
    },

    "BrandB_MidRange": {
        "brand_code": "BrandB",
        "tier": "Mid-range",
        "price_USD": 450,
        "volume_L": 55,
        "power_W": 1200,

        "claimed_dBA": 52.0,
        "measured_dBA": 53.8,

        "band_100Hz_dBA": 38.1,
        "band_125Hz_dBA": 39.5,
        "band_200Hz_dBA": 43.8,
        "band_250Hz_dBA": 44.2,
        "band_315Hz_dBA": 45.1,
        "band_500Hz_dBA": 42.3,
        "band_1kHz_dBA":  40.7,
        "band_2kHz_dBA":  38.1,
        "band_4kHz_dBA":  34.5,

        "dBA_convection_fan": 48.1,
        "dBA_magnetron": 43.3,
        "dBA_cooling_fan": 41.8,
        "dBA_transformer": 39.2,
        "dBA_door_seal": 35.5,

        "fan_blade_count": 9,
        "fan_type": "centrifugal",
        "has_magnetron_shield": True,
        "shield_mass_kg": 0.20,
        "panel_damping_pct": 40,
        "sound_pack": "PU_foam_15mm",
        "air_gap_mm": 15,
        "isolator_type": "rubber_mount",
        "isolator_stiffness_Npm": 15000,

        "IEC60704_limit_dBA": 58.0,
        "IEC60704_compliant": True,
        "energy_class": "A+",

        "launch_year": 2021,
        "market_share_pct": 18.5,
        "customer_noise_complaints_pct": 8.2,
        "return_rate_pct": 1.9,
    },

    "BrandC_Budget": {
        "brand_code": "BrandC",
        "tier": "Budget",
        "price_USD": 220,
        "volume_L": 30,
        "power_W": 900,

        "claimed_dBA": 55.0,
        "measured_dBA": 60.3,  # Exceeds own claim (common in budget)

        "band_100Hz_dBA": 44.1,
        "band_125Hz_dBA": 45.5,
        "band_200Hz_dBA": 49.8,
        "band_250Hz_dBA": 52.2,
        "band_315Hz_dBA": 53.1,
        "band_500Hz_dBA": 50.3,
        "band_1kHz_dBA":  48.7,
        "band_2kHz_dBA":  46.1,
        "band_4kHz_dBA":  42.5,

        "dBA_convection_fan": 55.1,
        "dBA_magnetron": 51.3,
        "dBA_cooling_fan": 48.8,
        "dBA_transformer": 47.2,
        "dBA_door_seal": 43.5,

        "fan_blade_count": 7,
        "fan_type": "axial_simple",
        "has_magnetron_shield": False,
        "shield_mass_kg": 0.0,
        "panel_damping_pct": 0,
        "sound_pack": "none",
        "air_gap_mm": 5,
        "isolator_type": "none",
        "isolator_stiffness_Npm": 200000,

        "IEC60704_limit_dBA": 58.0,
        "IEC60704_compliant": False,  # Exceeds limit
        "energy_class": "A",

        "launch_year": 2020,
        "market_share_pct": 22.3,
        "customer_noise_complaints_pct": 31.5,
        "return_rate_pct": 6.2,
    },

    "BrandD_BuiltIn": {
        "brand_code": "BrandD",
        "tier": "Built-in Premium",
        "price_USD": 1200,
        "volume_L": 45,
        "power_W": 1000,

        "claimed_dBA": 42.0,
        "measured_dBA": 43.5,

        "band_100Hz_dBA": 32.1,
        "band_125Hz_dBA": 33.5,
        "band_200Hz_dBA": 36.8,
        "band_250Hz_dBA": 37.2,
        "band_315Hz_dBA": 38.1,
        "band_500Hz_dBA": 36.3,
        "band_1kHz_dBA":  34.7,
        "band_2kHz_dBA":  32.1,
        "band_4kHz_dBA":  28.5,

        "dBA_convection_fan": 40.1,
        "dBA_magnetron": 36.3,
        "dBA_cooling_fan": 34.8,
        "dBA_transformer": 31.2,
        "dBA_door_seal": 28.5,

        "fan_blade_count": 13,
        "fan_type": "EC_backward_curved",
        "has_magnetron_shield": True,
        "shield_mass_kg": 0.40,
        "panel_damping_pct": 80,
        "sound_pack": "multi-layer_MLV+foam",
        "air_gap_mm": 28,
        "isolator_type": "gel_isolator",
        "isolator_stiffness_Npm": 1500,

        "IEC60704_limit_dBA": 58.0,
        "IEC60704_compliant": True,
        "energy_class": "A+++",

        "launch_year": 2023,
        "market_share_pct": 8.1,
        "customer_noise_complaints_pct": 1.2,
        "return_rate_pct": 0.5,
    },

    "BrandE_NewEntrant": {
        "brand_code": "BrandE",
        "tier": "Mid-premium",
        "price_USD": 520,
        "volume_L": 50,
        "power_W": 1100,

        "claimed_dBA": 49.0,
        "measured_dBA": 50.5,

        "band_100Hz_dBA": 36.1,
        "band_125Hz_dBA": 37.5,
        "band_200Hz_dBA": 41.8,
        "band_250Hz_dBA": 42.2,
        "band_315Hz_dBA": 43.1,
        "band_500Hz_dBA": 40.3,
        "band_1kHz_dBA":  38.7,
        "band_2kHz_dBA":  36.1,
        "band_4kHz_dBA":  32.5,

        "dBA_convection_fan": 46.1,
        "dBA_magnetron": 41.3,
        "dBA_cooling_fan": 39.8,
        "dBA_transformer": 37.2,
        "dBA_door_seal": 33.5,

        "fan_blade_count": 11,
        "fan_type": "centrifugal_optimized",
        "has_magnetron_shield": True,
        "shield_mass_kg": 0.28,
        "panel_damping_pct": 55,
        "sound_pack": "recycled_PET_20mm",
        "air_gap_mm": 20,
        "isolator_type": "rubber_composite",
        "isolator_stiffness_Npm": 8000,

        "IEC60704_limit_dBA": 58.0,
        "IEC60704_compliant": True,
        "energy_class": "A++",

        "launch_year": 2024,
        "market_share_pct": 5.8,
        "customer_noise_complaints_pct": 4.8,
        "return_rate_pct": 1.2,
    },
}


def generate_competitor_database():
    """Save competitor data as structured DataFrame."""
    print("  [BENCHMARK] Building competitor NVH database...")
    df = pd.DataFrame(COMPETITOR_DATA).T
    df.index.name = "product_id"
    df.to_csv(OUT / "competitor_data.csv")
    df.to_json(OUT / "competitor_data.json", orient="index", indent=2)
    print(f"  ✅ {len(df)} competitors in database")
    return df


def generate_benchmark_analysis(df_comp):
    """
    Gap analysis: where does OurUnit stand vs. competitors?
    Outputs structured gap report.
    """
    print("  [BENCHMARK] Running gap analysis...")

    our = df_comp.loc["OurUnit_MHC"]
    our_dba = float(our["measured_dBA"])
    target_dba = float(our["target_dBA"])

    records = []
    for prod_id, row in df_comp.iterrows():
        if prod_id == "OurUnit_MHC":
            continue
        comp_dba = float(row["measured_dBA"])
        gap = our_dba - comp_dba  # positive = we're louder (worse)
        records.append({
            "competitor": prod_id,
            "brand_code": row["brand_code"],
            "tier": row["tier"],
            "price_USD": float(row["price_USD"]),
            "competitor_dBA": comp_dba,
            "our_dBA": our_dba,
            "dBA_gap": round(gap, 1),
            "gap_direction": "We are LOUDER" if gap > 0 else "We are quieter",
            "target_dBA": target_dba,
            "dBA_to_match_competitor": round(our_dba - comp_dba, 1),
            "dBA_to_reach_target": round(our_dba - target_dba, 1),
            "competitor_shield_mass": float(row["shield_mass_kg"]),
            "competitor_damping_pct": float(row["panel_damping_pct"]),
            "competitor_air_gap_mm": float(row["air_gap_mm"]),
            "noise_complaint_gap_pct": float(our["customer_noise_complaints_pct"]) - float(row["customer_noise_complaints_pct"]),
        })

    df_gap = pd.DataFrame(records).sort_values("dBA_gap", ascending=False)
    df_gap.to_csv(OUT / "benchmark_gap_analysis.csv", index=False)

    # Key insights
    print("\n  ┌─ COMPETITIVE GAP ANALYSIS ─────────────────────────────")
    print(f"  │  Our baseline: {our_dba:.1f} dBA  |  Target: {target_dba:.1f} dBA")
    print("  ├────────────────────────────────────────────────────────")
    for _, r in df_gap.iterrows():
        status = "⚠️ " if r["dBA_gap"] > 0 else "✅"
        print(f"  │  {status} vs {r['brand_code']:<10}: gap = {r['dBA_gap']:+.1f} dBA  "
              f"(price: ${r['price_USD']:.0f}, tier: {r['tier']})")
    print("  └────────────────────────────────────────────────────────")

    return df_gap


def generate_feature_comparison(df_comp):
    """NVH feature matrix: what design features do competitors use?"""
    print("  [BENCHMARK] Generating feature comparison matrix...")

    features = [
        "fan_blade_count", "has_magnetron_shield", "shield_mass_kg",
        "panel_damping_pct", "sound_pack", "air_gap_mm",
        "isolator_type", "isolator_stiffness_Npm",
        "measured_dBA", "price_USD", "customer_noise_complaints_pct"
    ]

    df_feat = df_comp[features].copy()
    df_feat.index.name = "product_id"
    df_feat.to_csv(OUT / "feature_comparison_matrix.csv")

    # Compute best-in-class for each feature
    bic = {
        "fan_blade_count": df_comp["fan_blade_count"].max(),
        "shield_mass_kg": df_comp["shield_mass_kg"].max(),
        "panel_damping_pct": df_comp["panel_damping_pct"].max(),
        "air_gap_mm": df_comp["air_gap_mm"].max(),
        "measured_dBA": df_comp["measured_dBA"].min(),  # lower is better
    }

    print("\n  BEST-IN-CLASS NVH features across competitors:")
    for feat, val in bic.items():
        print(f"    {feat:<28}: {val}")

    return df_feat


def generate_market_positioning():
    """
    Market positioning: dBA vs. price scatter.
    Shows where target lands in market landscape.
    """
    print("  [BENCHMARK] Computing market positioning...")

    positioning = []
    for prod_id, data in COMPETITOR_DATA.items():
        positioning.append({
            "product": prod_id,
            "brand_code": data["brand_code"],
            "price_USD": data["price_USD"],
            "measured_dBA": data["measured_dBA"],
            "tier": data["tier"],
            "market_share_pct": data["market_share_pct"],
            "is_target": prod_id == "OurUnit_MHC",
        })

    # Add our target point
    positioning.append({
        "product": "OurUnit_TARGET",
        "brand_code": "MHC_target",
        "price_USD": COMPETITOR_DATA["OurUnit_MHC"]["price_USD"],  # same cost target
        "measured_dBA": 52.0,
        "tier": "Mid-premium (target)",
        "market_share_pct": None,
        "is_target": True,
    })

    df = pd.DataFrame(positioning)
    df.to_csv(OUT / "market_positioning.csv", index=False)

    print("\n  MARKET POSITIONING SUMMARY:")
    print(f"  {'Product':<25} {'dBA':>6}  {'Price':>7}  {'Tier'}")
    print("  " + "─" * 60)
    for _, r in df.sort_values("measured_dBA").iterrows():
        marker = " ← TARGET" if r["product"] == "OurUnit_TARGET" else ""
        marker2 = " ← CURRENT" if r["product"] == "OurUnit_MHC" else ""
        print(f"  {r['product']:<25} {float(r['measured_dBA']):>6.1f}  "
              f"${float(r['price_USD']):>6.0f}  {r['tier']}{marker}{marker2}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 62)
    print("  COMPETITIVE BENCHMARKING — Microwave/Convection Oven NVH")
    print("  Data source: Synthetic (based on published spec ranges)")
    print("═" * 62)

    df_comp  = generate_competitor_database()
    df_gap   = generate_benchmark_analysis(df_comp)
    df_feat  = generate_feature_comparison(df_comp)
    df_pos   = generate_market_positioning()

    print("\n  ✅ FILES SAVED:")
    for f in OUT.glob("*.csv"):
        print(f"    📄 {f.name}")
    for f in OUT.glob("*.json"):
        print(f"    📋 {f.name}")
    print("═" * 62 + "\n")
