"""
surrogate_model.py — Phase 3: ML Surrogate + Multi-objective Optimizer
=======================================================================
Trains RF & XGBoost surrogates on Phase-1 DOE data (4 active factors),
then runs NSGA-II to find the Pareto front: minimise dBA AND cost.

ESI VA One context:
  DOE training data was generated using physics equations calibrated
  against ESI VA One FEM/SEA aerovibro-acoustic simulation outputs for
  the 40-blade LH+RH axial blower at 3250 RPM. VA One provided:
    - Structural FEM panel responses (20–500 Hz, blower housing modes)
    - SEA airborne paths (500–8000 Hz, cavity absorption)
    - Receiver SPL at IEC 60704-1 reference point (1 m)
  Surrogate replaces repeated VA One runs during optimisation.

Active Design Variables (4):
  fan_blades [32,36,40,44,48], fan_rpm [2000–4500],
  panel_damp_pct [0–80 %], soundpack_dens [20–120 kg/m³]

Fixed (constrained — not in optimisation space):
  mag_shield_kg = 0.15 kg  |  air_gap_mm = 12 mm  |  isolator_K = 8000 N/m
"""

import numpy as np
import pandas as pd
import json, warnings
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

warnings.filterwarnings("ignore")

BASE   = Path(__file__).resolve().parent.parent
DATA   = BASE / "data"
SIMDIR = BASE / "simulation"
SIMDIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["fan_blades","fan_rpm","panel_damp_pct","soundpack_dens"]
TARGETS  = ["dBA","cost_index","weight_kg","thermal_index"]

BOUNDS_LO = np.array([32, 2000, 0,  20])
BOUNDS_HI = np.array([48, 4500, 80, 120])


def load_doe():
    doe = pd.read_csv(DATA / "doe_combined.csv")
    doe = doe.dropna(subset=FEATURES + TARGETS)
    return doe[FEATURES].copy(), doe[TARGETS].copy()


def prepare_data(X, y, test_size=0.2, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    scaler = StandardScaler()
    return Xtr, Xte, ytr, yte, scaler.fit_transform(Xtr), scaler.transform(Xte), scaler


def train_random_forest(Xtr, ytr, seed=42):
    return {col: RandomForestRegressor(n_estimators=300, max_features="sqrt",
            min_samples_leaf=3, random_state=seed, n_jobs=-1).fit(Xtr, ytr[col])
            for col in TARGETS}


def train_xgboost(Xtr_sc, ytr, seed=42):
    return {col: xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            verbosity=0).fit(Xtr_sc, ytr[col])
            for col in TARGETS}


def evaluate_models(rf, xgb_m, Xte, Xte_sc, yte):
    records = []
    for col in TARGETS:
        rp = rf[col].predict(Xte)
        xp = xgb_m[col].predict(Xte_sc)
        records.append({"target":col,
                         "RF_RMSE":round(np.sqrt(mean_squared_error(yte[col],rp)),4),
                         "RF_R2":round(r2_score(yte[col],rp),4),
                         "XGB_RMSE":round(np.sqrt(mean_squared_error(yte[col],xp)),4),
                         "XGB_R2":round(r2_score(yte[col],xp),4)})
    return pd.DataFrame(records)


def compute_shap(rf, Xtr, target="dBA"):
    expl = shap.TreeExplainer(rf[target])
    sv   = expl.shap_values(Xtr)
    df   = pd.DataFrame({"feature":FEATURES,
                          "mean_abs_shap":np.abs(sv).mean(axis=0)})
    df   = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


class OvenNVHProblem(Problem):
    """
    2-objective NSGA-II: minimise dBA + cost_index
    Constraints: weight_kg ≤ 1.0, thermal_index ≥ 0.75, dBA ≤ 52
    """
    def __init__(self, rf):
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=3,
                         xl=BOUNDS_LO, xu=BOUNDS_HI)
        self.rf = rf

    def _evaluate(self, X, out, *args, **kwargs):
        Xe = X.copy()
        # Round blade count to valid discrete values
        Xe[:,0] = np.clip(np.round(Xe[:,0]/4)*4, 32, 48)
        dBA_p    = self.rf["dBA"].predict(Xe)
        cost_p   = self.rf["cost_index"].predict(Xe)
        weight_p = self.rf["weight_kg"].predict(Xe)
        therm_p  = self.rf["thermal_index"].predict(Xe)
        out["F"] = np.column_stack([dBA_p, cost_p])
        out["G"] = np.column_stack([weight_p - 1.0,
                                     0.75 - therm_p,
                                     dBA_p - 52.0])


def run_nsga2(rf, pop_size=200, n_gen=150, seed=42):
    problem = OvenNVHProblem(rf)
    res = pymoo_minimize(problem, NSGA2(pop_size=pop_size),
                         get_termination("n_gen", n_gen),
                         seed=seed, verbose=False)
    if res.X is None or len(res.X) == 0:
        print("    ⚠  No feasible solutions — relaxing constraints")
        class Relaxed(OvenNVHProblem):
            def _evaluate(self, X, out, *args, **kwargs):
                Xe = X.copy(); Xe[:,0] = np.clip(np.round(Xe[:,0]/4)*4, 32, 48)
                out["F"] = np.column_stack([self.rf["dBA"].predict(Xe), self.rf["cost_index"].predict(Xe)])
                out["G"] = np.column_stack([self.rf["weight_kg"].predict(Xe)-2.0, 0.5-self.rf["thermal_index"].predict(Xe), self.rf["dBA"].predict(Xe)-55.0])
        res = pymoo_minimize(Relaxed(rf), NSGA2(pop_size=pop_size),
                             get_termination("n_gen", n_gen), seed=seed, verbose=False)

    pareto = pd.DataFrame(res.X, columns=FEATURES)
    pareto["dBA_pred"]        = res.F[:,0]
    pareto["cost_index_pred"] = res.F[:,1]
    pareto["dba_reduction"]   = 60.0 - pareto["dBA_pred"]
    pareto = pareto.sort_values("dBA_pred").reset_index(drop=True)

    feasible  = pareto[pareto["dBA_pred"] <= 52.0]
    if feasible.empty: feasible = pareto
    sweet = feasible.loc[feasible["cost_index_pred"].idxmin()].to_dict()
    return {"pareto_df": pareto, "sweet_spot": sweet}


def run_surrogate_pipeline():
    print("="*62)
    print("  MHC Oven — Surrogate + NSGA-II Optimizer  (Phase 3)")
    print("  Active factors: fan_blades, fan_rpm, panel_damp_pct, soundpack_dens")
    print("  Fixed : mag_shield=0.15kg | air_gap=12mm | isolator_K=8000 N/m")
    print("="*62)

    print("\n[A] Loading DOE (4 active factors)...")
    X, y = load_doe()
    print(f"    {len(X)} samples | {len(FEATURES)} features | {len(TARGETS)} targets")
    Xtr,Xte,ytr,yte,Xtr_sc,Xte_sc,scaler = prepare_data(X, y)

    print("\n[B] Training RF + XGBoost...")
    rf  = train_random_forest(Xtr, ytr)
    xgb_m = train_xgboost(Xtr_sc, ytr)

    print("\n[C] Model accuracy (hold-out)...")
    metrics = evaluate_models(rf, xgb_m, Xte, Xte_sc, yte)
    print(metrics.to_string(index=False))

    print("\n[D] SHAP feature importance (RF → dBA)...")
    shap_df = compute_shap(rf, Xtr)
    print(shap_df[["rank","feature","mean_abs_shap"]].to_string(index=False))

    print("\n[E] NSGA-II optimisation (200 pop × 150 gen)...")
    opt    = run_nsga2(rf)
    pareto = opt["pareto_df"]
    sweet  = opt["sweet_spot"]
    print(f"    Pareto: {len(pareto)} pts | dBA {pareto['dBA_pred'].min():.1f}–{pareto['dBA_pred'].max():.1f}")
    print(f"    Sweet spot: {sweet.get('dBA_pred',0):.1f} dBA @ cost={sweet.get('cost_index_pred',0):.3f}")

    metrics.to_csv(SIMDIR/"surrogate_metrics.csv", index=False)
    shap_df.to_csv(SIMDIR/"shap_importance.csv", index=False)
    pareto.to_csv(SIMDIR/"pareto_front.csv", index=False)
    with open(SIMDIR/"sweet_spot.json","w") as f:
        json.dump({k:float(v) for k,v in sweet.items()}, f, indent=2)
    print(f"\n✅  Phase-3 outputs → {SIMDIR}/")
    return {"rf_models":rf,"xgb_models":xgb_m,"scaler":scaler,
            "metrics":metrics,"shap_df":shap_df,"pareto_df":pareto,"sweet_spot":sweet}

if __name__ == "__main__":
    run_surrogate_pipeline()
