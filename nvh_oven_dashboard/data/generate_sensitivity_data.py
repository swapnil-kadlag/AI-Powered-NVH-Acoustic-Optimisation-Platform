"""
Sensitivity Analysis — MHC Oven
=====================================
Sobol / Morris / PCC over the 4 ACTIVE design variables.
Fixed factors (mag_shield_kg, air_gap_mm, isolator_K) excluded —
their contribution is treated as a constant baseline offset
calibrated from ESI VA One aerovibro-acoustic runs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from SALib.sample import saltelli, morris as morris_sample
from SALib.analyze import sobol, morris as morris_analyze

np.random.seed(42)
OUT = Path(__file__).parent

# Fixed values (from generate_doe_data.py)
MAG_SHIELD_KG = 0.15
AIR_GAP_MM    = 12.0
ISOLATOR_K    = 8000.0

PROBLEM = {
    "num_vars": 4,
    "names": ["fan_blades","fan_rpm","panel_damp_pct","soundpack_dens"],
    "bounds": [[32,48],[2000,4500],[0,80],[20,120]],
    "dists":  ["unif","unif","unif","unif"],
}

VAR_LABELS = {
    "fan_blades":     "Blade Count (LH+RH Blower)",
    "fan_rpm":        "Blower Speed (RPM)",
    "panel_damp_pct": "Panel CLD Coverage (%)",
    "soundpack_dens": "Sound Pack Density (kg/m³)",
}


def evaluate_dba(X):
    fan_blades     = X[:, 0]
    fan_rpm        = X[:, 1]
    panel_damp_pct = X[:, 2]
    soundpack_dens = X[:, 3]

    baseline = 60.0
    fan_noise_delta = 15 * np.log10(fan_rpm / 3250)
    blade_bonus  = np.where(fan_blades % 4 == 0, -1.5, 0.0)
    blade_spread = -1.5 * np.log10(fan_blades / 40)

    # Fixed-factor constant offsets (calibrated from VA One)
    mag_il  = 8.0 * np.log10(MAG_SHIELD_KG / 0.05)
    gap_il  = 2.0 * np.log10(AIR_GAP_MM / 5 + 1) * (1 - np.exp(-AIR_GAP_MM / 15))
    fn      = np.sqrt(ISOLATOR_K / 2.0) / (2 * np.pi)
    r       = 100 / (fn + 1e-9)
    iso_il  = float(4.0 * np.clip(1 - 1/max(r**2-1,1e-6), 0, 1)) if r > 1.41 else 0.0

    damp_il      = 3.5 * (1 - np.exp(-panel_damp_pct / 25))
    soundpack_il = 4.0 * np.log10(soundpack_dens / 20 + 1)

    return np.clip(baseline + fan_noise_delta + blade_bonus + blade_spread
                   - mag_il - damp_il - soundpack_il - gap_il - iso_il, 44, 64)


def evaluate_cost(X):
    fan_blades     = X[:, 0]
    panel_damp_pct = X[:, 2]
    soundpack_dens = X[:, 3]
    fan_cost       = 0.10 * (fan_blades - 32) / 16
    damp_cost      = 0.35 * panel_damp_pct / 80
    soundpack_cost = 0.35 * (soundpack_dens - 20) / 100
    return np.clip(fan_cost + damp_cost + soundpack_cost, 0, 1)


def compute_sobol_indices(n_samples=1024):
    print(f"  [SOBOL] N={n_samples}, total runs={n_samples*(2*4+2)}...")
    pv = saltelli.sample(PROBLEM, n_samples, calc_second_order=False)
    records = []
    for name, func in [("dBA", evaluate_dba), ("cost_index", evaluate_cost)]:
        Y  = func(pv)
        Si = sobol.analyze(PROBLEM, Y, calc_second_order=False, print_to_console=False)
        for i, var in enumerate(PROBLEM["names"]):
            records.append({"variable":var,"variable_label":VAR_LABELS[var],
                            "response":name,
                            "S1":round(max(Si["S1"][i],0),4),
                            "ST":round(max(Si["ST"][i],0),4),
                            "S1_conf":round(Si["S1_conf"][i],4),
                            "ST_conf":round(Si["ST_conf"][i],4)})
    df = pd.DataFrame(records)
    df.to_csv(OUT/"sobol_indices.csv", index=False)
    print(f"  ✅ Sobol indices: {len(df)} records")
    return df


def compute_morris_indices(n_trajectories=50):
    print(f"  [MORRIS] {n_trajectories} trajectories...")
    pv = morris_sample.sample(PROBLEM, N=n_trajectories, num_levels=8)
    records = []
    for Y, name in [(evaluate_dba(pv),"dBA"),(evaluate_cost(pv),"cost_index")]:
        Si = morris_analyze.analyze(PROBLEM, pv, Y, num_levels=8, print_to_console=False)
        for i, var in enumerate(PROBLEM["names"]):
            records.append({"variable":var,"variable_label":VAR_LABELS[var],
                            "response":name,"mu":round(Si["mu"][i],4),
                            "mu_star":round(Si["mu_star"][i],4),
                            "sigma":round(Si["sigma"][i],4)})
    df = pd.DataFrame(records)
    df.to_csv(OUT/"morris_indices.csv", index=False)
    print(f"  ✅ Morris indices: {len(df)} records")
    return df


def compute_pcc(n_samples=2000):
    print(f"  [PCC] {n_samples} MC samples...")
    X = np.column_stack([
        np.random.choice([32,36,40,44,48], n_samples).astype(float),
        np.random.uniform(2000, 4500, n_samples),
        np.random.uniform(0,    80,   n_samples),
        np.random.uniform(20,   120,  n_samples),
    ])
    Y_dba  = evaluate_dba(X)
    Y_cost = evaluate_cost(X)

    def partial_corr(X, Y):
        from numpy.linalg import lstsq
        pcc = []
        for j in range(X.shape[1]):
            others = [k for k in range(X.shape[1]) if k != j]
            Xo = np.column_stack([np.ones(len(X)), X[:, others]])
            rx = X[:,j] - Xo @ lstsq(Xo, X[:,j], rcond=None)[0]
            ry = Y      - Xo @ lstsq(Xo, Y,      rcond=None)[0]
            pcc.append(np.corrcoef(rx, ry)[0,1])
        return pcc

    records = []
    for i, var in enumerate(PROBLEM["names"]):
        records.append({"variable":var,"variable_label":VAR_LABELS[var],
                        "PCC_dBA":round(partial_corr(X,Y_dba)[i],4),
                        "PCC_cost":round(partial_corr(X,Y_cost)[i],4),
                        "abs_PCC_dBA":round(abs(partial_corr(X,Y_dba)[i]),4)})
    df = pd.DataFrame(records).sort_values("abs_PCC_dBA", ascending=False)
    df.to_csv(OUT/"pcc_results.csv", index=False)
    print(f"  ✅ PCC results: {len(df)} variables")
    return df


if __name__ == "__main__":
    print("\n" + "═"*62)
    print("  SENSITIVITY ANALYSIS — MHC Oven (4 active factors)")
    print("  Fixed: mag_shield=0.15kg | air_gap=12mm | isolator_K=8000 N/m")
    print("═"*62)
    df_s = compute_sobol_indices(512)
    df_m = compute_morris_indices(50)
    df_p = compute_pcc(2000)
    print("\n  TOP VARIABLES by Sobol ST (dBA):")
    for _, r in df_s[df_s["response"]=="dBA"].sort_values("ST",ascending=False).iterrows():
        print(f"    {r['variable_label']:<30} ST={r['ST']:.3f}")
    print("═"*62 + "\n")
