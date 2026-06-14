"""
NVH Measurement Data Generator — MHC Oven
==============================================
Generates physically plausible synthetic data comparable to real
Test.Lab / Sound Intensity scan measurements for a ~50L convection
oven (1200W, single fan, single magnetron).

Physics basis:
- Fan noise: BPF = N_blades × RPM/60, tonal + broadband
- Magnetron: 2×mains frequency (120 Hz) + harmonics
- Transformer: 100/200 Hz electromagnetic buzz
- SPL addition: L_total = 10·log10(Σ 10^(Li/10))

Author: NVH AI Dashboard
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Output directory ─────────────────────────────────────────────────────────
OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)

# ── 1/3 Octave Band Center Frequencies (Hz) ──────────────────────────────────
THIRD_OCT_FREQS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
    3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000
])

# ── Oven Operating Parameters ─────────────────────────────────────────────────
OVEN_PARAMS = {
    "power_W": 1200,
    "volume_L": 50,
    "fan_blades": 40,           # convection fan
    "fan_rpm": 3250,           # typical operating speed
    "cooling_fan_rpm": 2200,   # cooling fan faster, smaller
    "mains_freq_Hz": 50,       # 50 Hz (India/EU)
    "magnetron_freq_GHz": 2.45,
    "baseline_dBA": 60.0,
    "target_dBA": 52.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SOURCE SPL SPECTRA (1/3 Octave)
# ─────────────────────────────────────────────────────────────────────────────

def convection_fan_spectrum(freqs, n_blades=40, rpm=3250, seed_noise=0.8):
    """
    Convection fan noise model:
    - BPF = n_blades × rpm/60 = 2167 Hz for 40 blades @ 3250 RPM
    - Harmonics at 2×BPF, 3×BPF
    - Broadband turbulence: rolls off at -6 dB/oct above 2 kHz
    """
    bpf = n_blades * rpm / 60  # Blade Passage Frequency ~210 Hz

    # Broadband base: pink-noise shape (aerodynamic turbulence)
    broadband = 42 - 8 * np.log10(np.maximum(freqs / 200, 1))
    broadband = np.clip(broadband, 20, 52)

    # Tonal peaks at BPF harmonics
    spl = broadband.copy()
    for harmonic, level in [(1, 12), (2, 7), (3, 4)]:
        f_tone = harmonic * bpf
        # Add Gaussian peak around each harmonic
        peak = level * np.exp(-0.5 * ((np.log(freqs / f_tone) / 0.15) ** 2))
        spl += peak

    # Add measurement noise
    spl += np.random.normal(0, seed_noise, len(freqs))
    return np.clip(spl, 15, 65)


def magnetron_spectrum(freqs, mains_hz=50, seed_noise=0.6):
    """
    Magnetron electromagnetic noise:
    - Fundamental at 2×mains = 100 Hz (rectified mains drives magnetron)
    - Harmonics: 200, 400, 600, 800 Hz decaying
    - Broadband component is very low
    """
    spl = np.ones(len(freqs)) * 20  # low broadband floor

    # EM harmonics: 100, 200, 300... Hz
    for k, level in enumerate([15, 11, 8, 6, 4, 3], start=1):
        f_tone = k * 2 * mains_hz  # 100, 200, 300...
        peak = level * np.exp(-0.5 * ((np.log(np.maximum(freqs, 1) / f_tone) / 0.08) ** 2))
        spl = np.maximum(spl, spl + peak)

    spl += np.random.normal(0, seed_noise, len(freqs))
    return np.clip(spl, 10, 58)


def cooling_fan_spectrum(freqs, rpm=2200, seed_noise=0.9):
    """
    Cooling fan (axial, smaller, faster than convection fan):
    - BPF ~183 Hz (5 blades × 2200/60)
    - More broadband turbulence than convection fan
    - Higher frequency content due to higher RPM
    """
    bpf = 5 * rpm / 60  # ~183 Hz

    # Broadband: elevated from 500–4000 Hz range
    broadband = 38 - 5 * np.log10(np.maximum(freqs / 500, 0.5))
    broadband = np.clip(broadband, 18, 46)

    spl = broadband.copy()
    for harmonic, level in [(1, 8), (2, 5)]:
        f_tone = harmonic * bpf
        peak = level * np.exp(-0.5 * ((np.log(freqs / f_tone) / 0.18) ** 2))
        spl += peak

    spl += np.random.normal(0, seed_noise, len(freqs))
    return np.clip(spl, 12, 54)


def transformer_spectrum(freqs, mains_hz=50, seed_noise=0.5):
    """
    Transformer magnetostrictive buzz:
    - Fundamental at 2×mains = 100 Hz
    - Even harmonics: 200, 400 Hz dominant
    - Very tonal, narrow peaks
    """
    spl = np.ones(len(freqs)) * 18  # very low broadband

    for k, level in [(1, 13), (2, 10), (4, 7), (6, 4)]:
        f_tone = k * 2 * mains_hz
        peak = level * np.exp(-0.5 * ((np.log(np.maximum(freqs, 1) / f_tone) / 0.05) ** 2))
        spl = np.maximum(spl, spl + peak * 0.5)
        spl += peak * 0.5

    spl += np.random.normal(0, seed_noise, len(freqs))
    return np.clip(spl, 10, 55)


def door_seal_spectrum(freqs, seed_noise=0.7):
    """
    Door seal acoustic leakage radiation:
    - Resonance of air gap between door and frame
    - Broadband, peaks around 250–1000 Hz
    - Acts as secondary radiator for internal sources
    """
    # Resonance-like shape
    f_res = 400  # Hz, typical door cavity resonance
    broadband = 30 - 6 * np.log10(np.maximum(freqs / 100, 1))
    resonance = 8 * np.exp(-0.5 * ((np.log(freqs / f_res) / 0.4) ** 2))

    spl = broadband + resonance
    spl += np.random.normal(0, seed_noise, len(freqs))
    return np.clip(spl, 10, 48)


def compute_total_spl(spectra_list):
    """Logarithmic power addition: L_total = 10·log10(Σ 10^(Li/10))"""
    linear = np.sum([10 ** (s / 10) for s in spectra_list], axis=0)
    return 10 * np.log10(linear)


def spl_to_dba(spl_spectrum, freqs=THIRD_OCT_FREQS):
    """
    Apply A-weighting correction to 1/3 octave SPL spectrum.
    A-weighting formula per IEC 61672-1.
    """
    # A-weighting in dB at 1/3 octave centers (approximate tabulated values)
    a_weight = {
        20: -50.5, 25: -44.7, 31.5: -39.4, 40: -34.6, 50: -30.2,
        63: -26.2, 80: -22.5, 100: -19.1, 125: -16.1, 160: -13.4,
        200: -10.9, 250: -8.6, 315: -6.6, 400: -4.8, 500: -3.2,
        630: -1.9, 800: -0.8, 1000: 0.0, 1250: 0.6, 1600: 1.0,
        2000: 1.2, 2500: 1.3, 3150: 1.2, 4000: 1.0, 5000: 0.5,
        6300: -0.1, 8000: -1.1, 10000: -2.5, 12500: -4.3, 16000: -6.6
    }
    a_corr = np.array([a_weight.get(f, 0) for f in freqs])
    spl_a = spl_spectrum + a_corr
    # Overall dBA = 10·log10(Σ 10^(Li_A/10))
    return 10 * np.log10(np.sum(10 ** (spl_a / 10)))


def generate_source_spectra():
    """Generate 1/3 octave band SPL for each noise source + total."""
    print("  [1/3-OCT] Generating source spectra...")

    sources = {
        "convection_fan": convection_fan_spectrum(THIRD_OCT_FREQS),
        "magnetron":      magnetron_spectrum(THIRD_OCT_FREQS),
        "cooling_fan":    cooling_fan_spectrum(THIRD_OCT_FREQS),
        "transformer":    transformer_spectrum(THIRD_OCT_FREQS),
        "door_seal":      door_seal_spectrum(THIRD_OCT_FREQS),
    }

    # Total (logarithmic sum)
    sources["total"] = compute_total_spl(list(sources.values()))

    df = pd.DataFrame(sources, index=THIRD_OCT_FREQS)
    df.index.name = "freq_Hz"

    # Compute overall dBA per source
    dba_per_source = {}
    for src, spec in sources.items():
        dba_per_source[src] = spl_to_dba(spec)

    print(f"  Overall dBA per source:")
    for src, dba in dba_per_source.items():
        print(f"    {src:20s}: {dba:.1f} dBA")

    df.to_parquet(OUT / "source_spectra.parquet")
    df.to_csv(OUT / "source_spectra.csv")

    # Save dBA summary
    dba_df = pd.DataFrame.from_dict(
        dba_per_source, orient="index", columns=["overall_dBA"]
    )
    dba_df["contribution_pct"] = (
        10 ** (dba_df["overall_dBA"] / 10) /
        10 ** (dba_per_source["total"] / 10) * 100
    )
    dba_df.to_csv(OUT / "source_dba_summary.csv")

    print(f"  ✅ Total: {dba_per_source['total']:.1f} dBA (baseline target ~60 dBA)")
    return df, dba_per_source


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ODS (Operational Deflection Shape) DATA
# ─────────────────────────────────────────────────────────────────────────────

def generate_ods_data():
    """
    ODS: 12 sensor points on oven panels, measuring vibration amplitude
    at key frequencies. Realistic for a scan with accelerometers.

    Sensor layout (panel positions in mm from center):
    Top panel: 4 points, Side panels: 4 points, Rear: 2 points, Door: 2 points
    """
    print("  [ODS] Generating operational deflection shapes...")

    sensor_names = [
        "TOP_A", "TOP_B", "TOP_C", "TOP_D",   # Top panel
        "SIDE_L_A", "SIDE_L_B", "SIDE_R_A", "SIDE_R_B",  # Side panels
        "REAR_A", "REAR_B",                    # Rear wall
        "DOOR_A", "DOOR_B",                    # Door
    ]

    # Key frequencies of interest (BPF, transformer, magnetron)
    ods_freqs = [100, 120, 210, 420, 630, 1000, 2000]

    # Panel compliance (mm/N) — rear wall is stiffer, door is flexible
    panel_compliance = {
        "TOP": 1.2, "SIDE_L": 0.9, "SIDE_R": 0.85,
        "REAR": 0.5, "DOOR": 1.8
    }

    def get_compliance(sensor):
        for panel, comp in panel_compliance.items():
            if sensor.startswith(panel):
                return comp
        return 1.0

    data = []
    for freq in ods_freqs:
        row = {"freq_Hz": freq}
        for sensor in sensor_names:
            comp = get_compliance(sensor)
            # Amplitude in µm (micrometers): physics ~ F_excitation × compliance / freq²
            # At resonances (near 100, 2167 Hz), amplitudes are higher
            base_amp = comp * 50 / (1 + (freq / 300) ** 2)
            # Add spatial variation (mode shapes)
            spatial_factor = 1 + 0.3 * np.sin(np.pi * sensor_names.index(sensor) / 6)
            amplitude = base_amp * spatial_factor * (1 + np.random.normal(0, 0.1))
            row[f"{sensor}_um"] = round(abs(amplitude), 3)
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(OUT / "ods_data.csv", index=False)
    print(f"  ✅ ODS data: {len(sensor_names)} sensors × {len(ods_freqs)} frequencies")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — NTF (Noise Transfer Function)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ntf_data():
    """
    NTF: Noise Transfer Function (dB/N) from 5 structural excitation points
    to driver's ear position (1 m in front of oven, 1.5 m height).

    Excitation points:
    1. Fan mounting bracket
    2. Magnetron chassis bolt
    3. Transformer isolation pad
    4. Rear panel center
    5. Door latch mechanism

    Measured by impulse hammer excitation + microphone response.
    """
    print("  [NTF] Generating noise transfer functions...")

    excitation_points = [
        "fan_bracket",
        "magnetron_chassis",
        "transformer_pad",
        "rear_panel_center",
        "door_latch",
    ]

    ntf_freqs = THIRD_OCT_FREQS[4:]  # 50 Hz to 16 kHz

    # Base NTF shapes (resonances + anti-resonances typical for sheet metal)
    def ntf_shape(freqs, resonances, anti_resonances, base_level=-20):
        ntf = np.ones(len(freqs)) * base_level
        for f_res, gain in resonances:
            ntf += gain * np.exp(-0.5 * ((np.log(freqs / f_res) / 0.2) ** 2))
        for f_anti, loss in anti_resonances:
            ntf -= loss * np.exp(-0.5 * ((np.log(freqs / f_anti) / 0.15) ** 2))
        return ntf + np.random.normal(0, 0.5, len(freqs))

    ntf_configs = {
        "fan_bracket":        ([(210, 18), (630, 12), (1200, 8)],
                               [(400, 6), (800, 4)], -18),
        "magnetron_chassis":  ([(120, 22), (240, 15), (480, 10)],
                               [(300, 5), (600, 3)], -22),
        "transformer_pad":    ([(100, 20), (200, 14)],
                               [(150, 8), (350, 5)], -25),
        "rear_panel_center":  ([(315, 14), (800, 10)],
                               [(500, 4)], -20),
        "door_latch":         ([(400, 12), (1000, 8)],
                               [(630, 5)], -24),
    }

    data = {"freq_Hz": ntf_freqs}
    for point in excitation_points:
        res, anti_res, base = ntf_configs[point]
        data[f"NTF_{point}_dBperN"] = ntf_shape(ntf_freqs, res, anti_res, base)

    df = pd.DataFrame(data)
    df.to_csv(OUT / "ntf_data.csv", index=False)
    print(f"  ✅ NTF data: {len(excitation_points)} paths × {len(ntf_freqs)} frequencies")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — INSERTION LOSS (IL) of existing sound pack
# ─────────────────────────────────────────────────────────────────────────────

def generate_il_data():
    """
    Insertion Loss (IL) of existing baseline sound package.
    IL = SPL_without_treatment - SPL_with_treatment (higher = better).

    Measured at 10 representative 1/3 octave bands.
    Baseline sound pack: 20mm PU foam on rear/top panels.
    """
    print("  [IL] Generating insertion loss data...")

    il_freqs = [125, 250, 500, 1000, 2000, 4000, 8000, 250, 315, 400]
    il_freqs = [125, 250, 315, 400, 500, 630, 1000, 2000, 4000, 8000]

    # IL increases with frequency for foam materials
    # Low at low freq (mass law breakdown), peak in mid-range
    il_baseline = np.array([
        1.5, 2.8, 4.2, 5.8, 7.1, 8.4, 9.2, 10.1, 8.5, 6.2
    ])

    df = pd.DataFrame({
        "freq_Hz": il_freqs,
        "IL_existing_dB": il_baseline + np.random.normal(0, 0.3, len(il_freqs)),
        "IL_target_dB": il_baseline + 4.0,  # target: +4 dB improvement
        "IL_gap_dB": np.full(len(il_freqs), 4.0),
    })

    df.to_csv(OUT / "insertion_loss_data.csv", index=False)
    print(f"  ✅ IL data: {len(il_freqs)} frequency bands")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — OPERATING MODE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def generate_operating_modes():
    """
    NVH data across different operating modes:
    - Convection only (fan max, no magnetron)
    - Microwave only (magnetron on, cooling fan on)
    - Combined mode (convection + microwave simultaneously)
    - Standby (relay clicks, minor hum)
    """
    print("  [MODES] Generating operating mode data...")

    modes = {
        "convection_only": {
            "fan_speed_pct": 100, "magnetron_on": False,
            "overall_dBA": 54.2, "dominant_source": "convection_fan",
            "dominant_freq_Hz": 210
        },
        "microwave_only": {
            "fan_speed_pct": 60, "magnetron_on": True,
            "overall_dBA": 55.8, "dominant_source": "magnetron",
            "dominant_freq_Hz": 120
        },
        "combined_mode": {
            "fan_speed_pct": 100, "magnetron_on": True,
            "overall_dBA": 60.0, "dominant_source": "convection_fan",
            "dominant_freq_Hz": 210
        },
        "standby": {
            "fan_speed_pct": 0, "magnetron_on": False,
            "overall_dBA": 28.5, "dominant_source": "transformer",
            "dominant_freq_Hz": 100
        },
    }

    df = pd.DataFrame(modes).T
    df.index.name = "operating_mode"
    df.to_csv(OUT / "operating_modes.csv")
    print(f"  ✅ Operating modes: {len(modes)} modes")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  NVH MEASUREMENT DATA GENERATOR — MHC Oven")
    print("  Phase 1 — Synthetic Measurement Data")
    print("═" * 60)

    print("\n[STEP 1] Source 1/3 Octave Spectra")
    df_spectra, dba_summary = generate_source_spectra()

    print("\n[STEP 2] ODS (Operational Deflection Shapes)")
    df_ods = generate_ods_data()

    print("\n[STEP 3] NTF (Noise Transfer Functions)")
    df_ntf = generate_ntf_data()

    print("\n[STEP 4] Insertion Loss (Baseline Sound Pack)")
    df_il = generate_il_data()

    print("\n[STEP 5] Operating Modes")
    df_modes = generate_operating_modes()

    print("\n" + "─" * 60)
    print("  ✅ ALL MEASUREMENT DATA GENERATED")
    print(f"  Output directory: {OUT.resolve()}")
    print(f"  Files created:")
    for f in sorted(OUT.glob("*.csv")):
        print(f"    📄 {f.name}")
    for f in sorted(OUT.glob("*.parquet")):
        print(f"    📦 {f.name}")

    # Summary table
    print("\n  SOURCE RANKING (by dBA contribution):")
    print(f"  {'Source':<22} {'dBA':>6}  {'Share':>6}")
    print("  " + "-" * 38)
    sorted_sources = sorted(
        [(k, v) for k, v in dba_summary.items() if k != "total"],
        key=lambda x: x[1], reverse=True
    )
    for src, dba in sorted_sources:
        share = 10 ** (dba / 10) / 10 ** (dba_summary["total"] / 10) * 100
        print(f"  {src:<22} {dba:>6.1f}  {share:>5.1f}%")
    print(f"  {'TOTAL':<22} {dba_summary['total']:>6.1f}  100.0%")
    print("═" * 60 + "\n")
