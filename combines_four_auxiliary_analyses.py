#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
COMBINED AUXILIARY ANALYSES (EXACT c₀ DEFINITION)
===============================================================================

This script combines four auxiliary analyses using the exact c₀ definition:
  1. c₀ vs. Galactocentric Radius (outlier concentration plot)
  2. Tautology Test (spurious convergence check)
  3. Theoretical RAR Comparison (causal vs. MOND at low accelerations)
  4. Geometric Cut Test (R > 2 kpc cut)

All analyses use the EXACT definition: c₀ = v_bar / sqrt(e^{2φ} - 1)

Author  : Mahmoud F. Abdel-Sattar
Date    : 2026
Version : 3.0 (Exact c₀, combined)
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.stats import lognorm

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Data file (output from main EXACT c₀ analysis)
    SPARC_CSV: str = "sparc_complete_analysis_exact_c0/all_measurements_20260612_042844.csv"
    
    # Physical constants
    C0_PREDICTED: float = 104.3
    A0_MOND: float = 1.2e-10
    PHI_MAX: float = 0.429
    AMPLIFICATION: float = np.exp(2 * PHI_MAX)  # 2.357

    # Quality cuts
    C0_LOW: float = 50.0
    C0_HIGH: float = 200.0
    RADIUS_CUT: float = 2.0  # kpc

    # Output
    FIGURE_DPI: int = 300
    RANDOM_SEED: int = 42
    OUTPUT_DIR: str = "combined_auxiliary_exact_c0"


# =============================================================================
# C₀ CALCULATOR (EXACT DEFINITION)
# =============================================================================

class C0Calculator:
    """Computes c₀ using the EXACT definition: c₀ = v_bar / sqrt(e^{2φ} - 1)."""
    
    @staticmethod
    def compute(v_obs: np.ndarray, v_bar: np.ndarray) -> np.ndarray:
        v_bar_safe = np.maximum(v_bar, 0.1)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = np.log(np.maximum(v_obs, 1e-6) / v_bar_safe)
            exp2phi = np.exp(2.0 * phi)
            denominator = np.sqrt(np.maximum(exp2phi - 1.0, 1e-12))
            # EXACT: v_bar / denominator
            c0 = np.where(phi > 0, v_bar_safe / denominator, np.nan)
        return c0


# =============================================================================
# ANALYSIS 1: c₀ vs. GALACTOCENTRIC RADIUS
# =============================================================================

def analysis_1_c0_vs_radius(output_dir: Path):
    """Generate figure showing excluded points concentrated at small radii."""
    print("\n" + "="*70)
    print("ANALYSIS 1: c₀ vs. Galactocentric Radius")
    print("="*70)
    
    df = pd.read_csv(Config.SPARC_CSV)
    c0 = df["c0_corrected"].values
    radius = df["radius_kpc"].values
    
    mask_low = c0 < Config.C0_LOW
    mask_high = c0 > Config.C0_HIGH
    mask_outlier = mask_low | mask_high
    mask_good = ~mask_outlier
    
    print(f"Total points: {len(c0)}")
    print(f"Good points (50–200 km/s): {np.sum(mask_good)}")
    print(f"Outliers (c₀ < 50): {np.sum(mask_low)}")
    print(f"Outliers (c₀ > 200): {np.sum(mask_high)}")
    
    plt.rcParams.update({'font.family':'serif','font.size':12,'axes.labelsize':14,'axes.titlesize':14,'legend.fontsize':11})
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.axhspan(0, Config.C0_LOW, alpha=0.08, color='red', zorder=0)
    ax.axhspan(Config.C0_HIGH, 350, alpha=0.08, color='red', zorder=0)
    
    ax.scatter(radius[mask_good], c0[mask_good], alpha=0.25, s=6, color='steelblue',
               label=f'Retained ($N={np.sum(mask_good)}$)', zorder=2)
    ax.scatter(radius[mask_low], c0[mask_low], alpha=0.50, s=12, color='red', marker='o',
               label=f'$c_0 < {Config.C0_LOW:.0f}$ ($N={np.sum(mask_low)}$)', zorder=3)
    ax.scatter(radius[mask_high], c0[mask_high], alpha=0.50, s=12, color='darkred', marker='s',
               label=f'$c_0 > {Config.C0_HIGH:.0f}$ ($N={np.sum(mask_high)}$)', zorder=3)
    
    ax.axhline(Config.C0_LOW, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(Config.C0_HIGH, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(Config.C0_PREDICTED, color='green', linestyle='-', linewidth=2,
               label=f'$c_0 = {Config.C0_PREDICTED:.1f}$ km/s')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Galactocentric Radius $R$ [kpc]')
    ax.set_ylabel(r'Characteristic Speed $c_0$ [km s$^{-1}$]')
    ax.set_xlim(0.08, 120)
    ax.set_ylim(0, 350)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    textstr = (f'Retained: {np.sum(mask_good)}/{len(c0)} ({100*np.sum(mask_good)/len(c0):.1f}%)\n'
               f'Excluded (low): {np.sum(mask_low)} ({100*np.sum(mask_low)/len(c0):.1f}%)\n'
               f'Excluded (high): {np.sum(mask_high)} ({100*np.sum(mask_high)/len(c0):.1f}%)')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = output_dir / 'figure_c0_vs_radius_outliers.pdf'
    plt.savefig(filename, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


# =============================================================================
# ANALYSIS 2: TAUTOLOGY TEST (EXACT c₀)
# =============================================================================

class SyntheticDataGenerator:
    def __init__(self, df_sparc: pd.DataFrame):
        self.df_sparc = df_sparc.copy()
        self.n_points = len(df_sparc)
        self.rng = np.random.default_rng(Config.RANDOM_SEED)

    def generate_dark_halo(self, halo_scale: float = 100.0) -> pd.DataFrame:
        df = self.df_sparc.copy()
        v_halo = lognorm.rvs(s=0.6, scale=halo_scale, size=self.n_points, random_state=self.rng)
        df['v_obs_syn'] = np.sqrt(df['v_bar']**2 + v_halo**2)
        return df

    @staticmethod
    def _mond_nu(g_bar: np.ndarray) -> np.ndarray:
        y = g_bar / Config.A0_MOND
        return 0.5 + np.sqrt(0.25 + 1.0 / np.maximum(y, 1e-12))

    def generate_mond(self) -> pd.DataFrame:
        df = self.df_sparc.copy()
        r_m = df['radius_kpc'].values * 3.086e19
        v_bar_ms = df['v_bar'].values * 1e3
        g_bar = np.maximum(v_bar_ms**2 / r_m, 1e-16)
        nu = self._mond_nu(g_bar)
        v_obs_ms = v_bar_ms * np.sqrt(nu)
        df['v_obs_syn'] = v_obs_ms / 1e3
        return df

    def generate_randomised(self) -> pd.DataFrame:
        df = self.df_sparc.copy()
        df['v_obs_syn'] = self.rng.permutation(df['v_obs'].values)
        return df


class TautologyTest:
    def __init__(self, df_sparc: pd.DataFrame):
        self.df_real = df_sparc.copy()
        self.generator = SyntheticDataGenerator(self.df_real)
        self.c0_real = C0Calculator.compute(self.df_real['v_obs'].values, self.df_real['v_bar'].values)
        self.c0_dark = None
        self.c0_mond = None
        self.c0_rand = None
        self.results: Dict[str, Dict[str, float]] = {}

    def _apply_cut(self, c0: np.ndarray) -> np.ndarray:
        return c0[(c0 >= Config.C0_LOW) & (c0 <= Config.C0_HIGH)]

    def _stats(self, c0: np.ndarray) -> Dict[str, float]:
        c0_clean = self._apply_cut(c0)
        n = len(c0_clean)
        if n < 10:
            return {'N': n, 'mean': np.nan, 'std': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
        rng = np.random.default_rng(Config.RANDOM_SEED)
        boots = [np.mean(rng.choice(c0_clean, size=n, replace=True)) for _ in range(10_000)]
        boots = np.array(boots)
        return {'N': n, 'mean': np.mean(c0_clean), 'std': np.std(c0_clean),
                'ci_low': np.percentile(boots, 2.5), 'ci_high': np.percentile(boots, 97.5)}

    def run(self) -> Dict[str, Dict[str, float]]:
        print("\n" + "="*70)
        print("ANALYSIS 2: TAUTOLOGY TEST (EXACT c₀)")
        print("="*70)
        
        self.results['SPARC (clean)'] = self._stats(self.c0_real)
        r = self.results['SPARC (clean)']
        print(f"  SPARC real: N={r['N']}, mean={r['mean']:.1f}, std={r['std']:.1f} km/s")
        
        df_dark = self.generator.generate_dark_halo()
        self.c0_dark = C0Calculator.compute(df_dark['v_obs_syn'].values, df_dark['v_bar'].values)
        self.results['Dark-halo'] = self._stats(self.c0_dark)
        r = self.results['Dark-halo']
        print(f"  Dark-halo:  N={r['N']}, mean={r['mean']:.1f}, std={r['std']:.1f} km/s")
        
        df_mond = self.generator.generate_mond()
        self.c0_mond = C0Calculator.compute(df_mond['v_obs_syn'].values, df_mond['v_bar'].values)
        self.results['MOND'] = self._stats(self.c0_mond)
        r = self.results['MOND']
        print(f"  MOND:       N={r['N']}, mean={r['mean']:.1f}, std={r['std']:.1f} km/s")
        
        df_rand = self.generator.generate_randomised()
        self.c0_rand = C0Calculator.compute(df_rand['v_obs_syn'].values, df_rand['v_bar'].values)
        self.results['Randomised'] = self._stats(self.c0_rand)
        r = self.results['Randomised']
        print(f"  Randomised: N={r['N']}, mean={r['mean']:.1f}, std={r['std']:.1f} km/s")
        
        print("=" * 70)
        return self.results

    def plot_comparison(self, output_path: str) -> None:
        if self.c0_dark is None:
            raise RuntimeError("Run the test first.")
        fig, ax = plt.subplots(figsize=(10, 7))
        datasets = [
            ('SPARC (clean)', self.c0_real, 'steelblue'),
            ('Dark-halo', self.c0_dark, 'darkorange'),
            ('MOND', self.c0_mond, 'darkgreen'),
            ('Randomised', self.c0_rand, 'crimson')
        ]
        for label, c0_arr, color in datasets:
            c0_cut = self._apply_cut(c0_arr)
            if len(c0_cut) > 10:
                ax.hist(c0_cut, bins=40, alpha=0.4, color=color, density=True,
                        label=f"{label} (σ={np.std(c0_cut):.1f})")
        ax.set_xlabel(r'$c_0$ [km s$^{-1}$]')
        ax.set_ylabel('Probability density')
        ax.set_title('Test for Spurious Convergence (Exact c₀)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure saved: {output_path}")

    def generate_report(self, output_dir: Path) -> None:
        out = output_dir
        out.mkdir(parents=True, exist_ok=True)
        lines = [r"\begin{table}[htbp]", r"\centering",
                 r"\caption{Comparison of $c_0$ distributions from SPARC and synthetic datasets (Exact $c_0$).}",
                 r"\label{tab:tautology}", r"\begin{tabular}{l c c c c}", r"\hline",
                 r"Scenario & $N$ & $\langle c_0 \rangle$ [km/s] & $\sigma_{c_0}$ [km/s] & 95\% CI [km/s] \\", r"\hline"]
        for label, s in self.results.items():
            lines.append(f"{label} & {s['N']} & {s['mean']:.1f} & {s['std']:.1f} & [{s['ci_low']:.1f}, {s['ci_high']:.1f}] \\\\")
        lines.append(r"\hline"); lines.append(r"\end{tabular}"); lines.append(r"\end{table}")
        with open(out / "tautology_table.tex", 'w') as f:
            f.write('\n'.join(lines))
        print(f"✓ LaTeX table saved: {out / 'tautology_table.tex'}")


def analysis_2_tautology_test(output_dir: Path):
    df_sparc = pd.read_csv(Config.SPARC_CSV)
    test = TautologyTest(df_sparc)
    test.run()
    test.plot_comparison(str(output_dir / 'figure_tautology_test.pdf'))
    test.generate_report(output_dir)
    print("✓ Tautology test complete.")


# =============================================================================
# ANALYSIS 3: THEORETICAL RAR COMPARISON
# =============================================================================

def analysis_3_rar_theoretical(output_dir: Path):
    """Generate RAR theoretical comparison plot."""
    print("\n" + "="*70)
    print("ANALYSIS 3: Theoretical RAR Comparison")
    print("="*70)
    
    df = pd.read_csv(Config.SPARC_CSV)
    c0_clean = df['c0_corrected'].values
    mask_clean = (c0_clean >= Config.C0_LOW) & (c0_clean <= Config.C0_HIGH)
    df_clean = df[mask_clean]
    print(f"Clean sample points: {len(df_clean)}")
    
    g_bar_range = np.logspace(-14, -8, 200)
    g_obs_unity = g_bar_range
    g_obs_causal = Config.AMPLIFICATION * g_bar_range
    y = g_bar_range / Config.A0_MOND
    nu_mond = 0.5 + np.sqrt(0.25 + 1.0 / np.maximum(y, 1e-12))
    g_obs_mond = g_bar_range * nu_mond
    
    plt.rcParams.update({'font.family':'serif','font.size':12})
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df_clean['g_N'].values, df_clean['g_obs'].values, alpha=0.15, s=4, color='gray',
               label=f'SPARC clean sample (N = {len(df_clean)})', zorder=1)
    ax.plot(g_bar_range, g_obs_unity, 'k--', linewidth=1.5, label='$g_{\\mathrm{obs}} = g_{\\mathrm{bar}}$')
    ax.plot(g_bar_range, g_obs_causal, 'r-', linewidth=2.5,
            label=f'Causal: $g_{{\\mathrm{{obs}}}} = {Config.AMPLIFICATION:.3f}\\,g_{{\\mathrm{{bar}}}}$')
    ax.plot(g_bar_range, g_obs_mond, 'b-.', linewidth=2.5,
            label=f'MOND: $a_0 = {Config.A0_MOND:.1e}$ m s$^{{-2}}$')
    ax.axvspan(1e-14, 1e-12, alpha=0.07, color='orange')
    ax.text(2e-13, 3e-12, 'Low-acceleration\nregime\n(future probes)', fontsize=9,
            ha='center', color='darkorange', style='italic')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$g_{\mathrm{bar}}$ [m s$^{-2}$] (baryonic)')
    ax.set_ylabel(r'$g_{\mathrm{obs}}$ [m s$^{-2}$] (observed)')
    ax.set_title('Radial Acceleration Relation: Causal Field Model vs. MOND')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-14, 1e-8); ax.set_ylim(1e-14, 1e-8)
    plt.tight_layout()
    
    filename = output_dir / 'figure_RAR_theoretical_comparison.pdf'
    plt.savefig(filename, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


# =============================================================================
# ANALYSIS 4: GEOMETRIC CUT TEST (R > 2 kpc)
# =============================================================================

def analysis_4_radius_cut(output_dir: Path):
    """Test c₀ with R > 2 kpc cut."""
    print("\n" + "="*70)
    print("ANALYSIS 4: Geometric Cut (R > 2 kpc)")
    print("="*70)
    
    df = pd.read_csv(Config.SPARC_CSV)
    mask_radius = df['radius_kpc'] >= Config.RADIUS_CUT
    df_cut = df[mask_radius]
    c0_radius_cut = df_cut['c0_corrected'].values
    
    n_total = len(df)
    n_cut = len(df_cut)
    print(f"Total points: {n_total}")
    print(f"Points with R >= {Config.RADIUS_CUT} kpc: {n_cut} ({100*n_cut/n_total:.1f}%)")
    
    mean_c0 = np.mean(c0_radius_cut)
    std_c0 = np.std(c0_radius_cut)
    
    rng = np.random.default_rng(Config.RANDOM_SEED)
    boot_means = [np.mean(rng.choice(c0_radius_cut, size=n_cut, replace=True)) for _ in range(10_000)]
    boot_means = np.array(boot_means)
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    
    print(f"\nRESULTS FOR R >= {Config.RADIUS_CUT} kpc CUT:")
    print(f"  Mean c₀ = {mean_c0:.1f} ± {std_c0:.1f} km/s")
    print(f"  95% CI  = [{ci_low:.1f}, {ci_high:.1f}] km/s")
    print(f"  Contains {Config.C0_PREDICTED}? {'YES' if ci_low <= Config.C0_PREDICTED <= ci_high else 'NO'}")
    
    # Save result to text file
    with open(output_dir / 'radius_cut_results.txt', 'w') as f:
        f.write(f"R > {Config.RADIUS_CUT} kpc cut results (Exact c₀):\n")
        f.write(f"N = {n_cut} / {n_total} ({100*n_cut/n_total:.1f}%)\n")
        f.write(f"Mean c₀ = {mean_c0:.1f} ± {std_c0:.1f} km/s\n")
        f.write(f"95% CI = [{ci_low:.1f}, {ci_high:.1f}] km/s\n")
        f.write(f"Contains {Config.C0_PREDICTED}? {'YES' if ci_low <= Config.C0_PREDICTED <= ci_high else 'NO'}\n")
    print(f"✓ Results saved: {output_dir / 'radius_cut_results.txt'}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMBINED AUXILIARY ANALYSES (EXACT c₀)")
    print("="*70)
    
    # Run all four analyses
    analysis_1_c0_vs_radius(output_dir)
    analysis_2_tautology_test(output_dir)
    analysis_3_rar_theoretical(output_dir)
    analysis_4_radius_cut(output_dir)
    
    print("\n" + "="*70)
    print("✓ ALL AUXILIARY ANALYSES COMPLETE")
    print(f"  Results saved in: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()