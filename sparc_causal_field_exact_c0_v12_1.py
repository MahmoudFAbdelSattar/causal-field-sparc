#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
SPARC DATA ANALYSIS FOR THE CAUSAL FIELD MODEL (EXACT c₀ DEFINITION)
===============================================================================

This module performs a complete analysis of the SPARC galaxy sample using
the EXACT definition of the characteristic speed:
    c₀ = v_bar / sqrt(e^{2φ} - 1) = v_obs * e^{-φ} / sqrt(e^{2φ} - 1)

Author  : Mahmoud F. Abdel-Sattar
Email   : m.f.abdel-sattar@azhar.edu.eg
Date    : 2026
Version : 12.1 (Exact c₀ definition, revised figures and classification)
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from tqdm import tqdm
import requests
import warnings
from datetime import datetime
from pathlib import Path
import os
import sys

# Optional MCMC libraries
try:
    import pymc as pm
    import arviz as az
    MCMC_AVAILABLE = True
    print("✅ PyMC and ArviZ are available.")
except ImportError:
    MCMC_AVAILABLE = False
    print("⚠️ PyMC/ArviZ not installed. Bayesian analysis will be skipped.")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    SPARC_URL = "https://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt"
    C0_PREDICTED = 104.3          # km/s (predicted universal constant)
    C_LIGHT = 299792.458           # km/s (speed of light in vacuum)
    PHI_MAX = 0.429                # maximum causal field (from complete theory)

    MIN_V_BAR = 0.1
    MAX_PHI = 2.5
    MIN_C0 = 30.0
    MAX_C0 = 300.0
    MIN_POINTS_PER_GALAXY = 2

    # Quality cut window for clean sample (used in figure c0 vs radius)
    C0_LOW = 50.0
    C0_HIGH = 200.0

    PROGRESSIVE_CUTS = [
        ("No cut", 0, 300),
        ("Moderate cut", 50, 200),
        ("Strict cut", 70, 150),
        ("Very strict", 80, 130),
        ("Extremely strict", 85, 125)
    ]

    BOOTSTRAP_N_SAMPLES = 10000
    BOOTSTRAP_CONFIDENCE = 0.95

    MCMC_N_SAMPLES = 2000
    MCMC_N_TUNE = 1000
    MCMC_N_CHAINS = 4

    SIMULATION_N_SIM = 10000
    SIMULATION_DEFAULT_SIGMA_VOBS = 10
    SIMULATION_DEFAULT_SIGMA_VBAR = 15

    USE_MEASURED_ERRORS = True
    MIN_POINTS_FOR_ERROR_EST = 10

    FIGURE_DPI = 300
    OUTPUT_DIR = "sparc_complete_analysis_exact_c0"
    RANDOM_SEED = 42

    RADIUS_BINS = [0, 1, 5, 10, 50, 100]
    PHI_BINS = np.arange(0, 1.1, 0.2)


# =============================================================================
# DATA ACQUISITION
# =============================================================================

class SPARCDataFetcher:
    @staticmethod
    def fetch_data():
        print("\n" + "="*80)
        print("DATA ACQUISITION")
        print("="*80)
        print(f"Downloading from: {Config.SPARC_URL}")
        try:
            response = requests.get(Config.SPARC_URL, timeout=120, verify=False)
            response.raise_for_status()
            print(f"✓ Successfully downloaded {len(response.text):,} bytes")
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"✗ Download failed: {e}")
            raise

    @staticmethod
    def parse_raw_data(raw_text):
        print("\nParsing SPARC data...")
        lines = raw_text.strip().split('\n')
        galaxies = {}
        for line in tqdm(lines, desc="Processing lines"):
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            galaxy = parts[0]
            try:
                radius = float(parts[2])
                v_obs = abs(float(parts[3]))
                v_gas = abs(float(parts[5]))
                v_disk = abs(float(parts[6]))
                v_bulge = abs(float(parts[7]) if len(parts) > 7 else 0.0)
                if galaxy not in galaxies:
                    galaxies[galaxy] = []
                galaxies[galaxy].append({
                    'radius': radius,
                    'v_obs': v_obs,
                    'v_gas': v_gas,
                    'v_disk': v_disk,
                    'v_bulge': v_bulge
                })
            except (ValueError, IndexError):
                continue
        total_points = sum(len(g) for g in galaxies.values())
        print(f"✓ Parsed {len(galaxies)} galaxies with {total_points} total data points")
        return galaxies


# =============================================================================
# CORE CALCULATIONS (EXACT c₀ DEFINITION)
# =============================================================================

class CausalFieldCalculator:
    @staticmethod
    def baryonic_velocity(v_gas, v_disk, v_bulge):
        return np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)

    @staticmethod
    def causal_field(v_obs, v_bar):
        if v_obs <= v_bar or v_bar <= 0:
            return np.nan
        return np.log(v_obs / v_bar)

    @staticmethod
    def characteristic_speed_corrected(v_obs, phi):
        """
        EXACT definition:
        c₀ = v_bar / sqrt(e^{2φ} - 1) = v_obs * e^{-φ} / sqrt(e^{2φ} - 1)
        """
        if phi <= 0 or not np.isfinite(phi):
            return np.nan
        denominator = np.sqrt(np.exp(2 * phi) - 1)
        if denominator <= 0:
            return np.nan
        v_bar_equiv = v_obs * np.exp(-phi)
        return v_bar_equiv / denominator

    @staticmethod
    def accelerations(v_obs, v_bar, radius_kpc):
        r_m = radius_kpc * 3.086e19
        v_obs_ms = v_obs * 1000
        v_bar_ms = v_bar * 1000
        g_obs = v_obs_ms**2 / r_m
        g_N = v_bar_ms**2 / r_m
        return g_obs, g_N

    @staticmethod
    def c0_error_propagation(v_obs, v_bar, phi, sigma_vobs, sigma_vbar):
        """
        Error propagation for exact c₀ = v_bar / sqrt(e^{2φ} - 1).
        """
        if phi <= 0 or not np.isfinite(phi):
            return np.nan
        exp2phi = np.exp(2 * phi)
        denom = np.sqrt(exp2phi - 1)
        # dc₀/dv_bar
        dc0_dvbar = 1.0 / denom
        # dc₀/dφ
        dc0_dphi = -v_bar * exp2phi / (denom**3)
        # dφ/dv_obs, dφ/dv_bar
        dphi_dvobs = 1.0 / v_obs
        dphi_dvbar = -1.0 / v_bar
        term1 = (dc0_dvbar**2) * sigma_vbar**2
        term2 = (dc0_dphi**2) * (dphi_dvobs**2 * sigma_vobs**2 + dphi_dvbar**2 * sigma_vbar**2)
        return np.sqrt(term1 + term2)


# =============================================================================
# ERROR ESTIMATOR
# =============================================================================

class ErrorEstimator:
    def __init__(self):
        self.galaxy_vobs_errors = {}
        self.galaxy_vbar_errors = {}
        self.global_vobs_error = None
        self.global_vbar_error = None

    def estimate_errors_from_galaxy(self, galaxy_name, data_points):
        if len(data_points) < Config.MIN_POINTS_FOR_ERROR_EST:
            return None, None
        sorted_points = sorted(data_points, key=lambda x: x['radius'])
        v_obs_vals, v_bar_vals = [], []
        for point in sorted_points:
            v_bar = CausalFieldCalculator.baryonic_velocity(
                point['v_gas'], point['v_disk'], point['v_bulge'])
            if v_bar < Config.MIN_V_BAR:
                continue
            phi = CausalFieldCalculator.causal_field(point['v_obs'], v_bar)
            if np.isfinite(phi) and phi <= Config.MAX_PHI:
                v_obs_vals.append(point['v_obs'])
                v_bar_vals.append(v_bar)
        if len(v_obs_vals) < Config.MIN_POINTS_FOR_ERROR_EST:
            return None, None
        v_obs_vals = np.array(v_obs_vals)
        v_bar_vals = np.array(v_bar_vals)
        sigma_vobs = max(1.4826 * np.median(np.abs(v_obs_vals - np.median(v_obs_vals))), 0.1)
        sigma_vbar = max(1.4826 * np.median(np.abs(v_bar_vals - np.median(v_bar_vals))), 0.1)
        return sigma_vobs, sigma_vbar

    def estimate_global_errors(self, all_galaxies):
        all_vobs_err, all_vbar_err = [], []
        for name, pts in all_galaxies.items():
            sv, sb = self.estimate_errors_from_galaxy(name, pts)
            if sv is not None:
                self.galaxy_vobs_errors[name] = sv
                self.galaxy_vbar_errors[name] = sb
                all_vobs_err.append(sv)
                all_vbar_err.append(sb)
        if all_vobs_err:
            self.global_vobs_error = np.median(all_vobs_err)
            self.global_vbar_error = np.median(all_vbar_err)
            print(f"\n📊 Estimated global errors:")
            print(f"   σ_vobs = {self.global_vobs_error:.2f} km/s")
            print(f"   σ_vbar = {self.global_vbar_error:.2f} km/s")
        return self.global_vobs_error, self.global_vbar_error


# =============================================================================
# GALAXY ANALYSIS
# =============================================================================

class GalaxyAnalyzer:
    def __init__(self, error_estimator=None):
        self.calculator = CausalFieldCalculator()
        self.error_estimator = error_estimator
        self.all_c0 = []
        self.all_phi = []
        self.all_vobs = []
        self.all_vbar = []
        self.all_radii = []
        self.all_g_obs = []
        self.all_g_N = []
        self.all_c0_errors = []
        self.galaxy_results = {}
        self.galaxies_with_valid_c0 = set()

    def process_galaxy(self, galaxy_name, data_points):
        galaxy_c0, galaxy_phi, galaxy_vobs, galaxy_vbar = [], [], [], []
        galaxy_radii, galaxy_g_obs, galaxy_g_N, galaxy_c0_errors = [], [], [], []
        valid_points = []
        sigma_vobs = sigma_vbar = None
        if self.error_estimator:
            sigma_vobs = self.error_estimator.galaxy_vobs_errors.get(galaxy_name)
            sigma_vbar = self.error_estimator.galaxy_vbar_errors.get(galaxy_name)
        for point in data_points:
            v_bar = self.calculator.baryonic_velocity(point['v_gas'], point['v_disk'], point['v_bulge'])
            if v_bar < Config.MIN_V_BAR:
                continue
            phi = self.calculator.causal_field(point['v_obs'], v_bar)
            if not np.isfinite(phi) or phi > Config.MAX_PHI:
                continue
            c0 = self.calculator.characteristic_speed_corrected(point['v_obs'], phi)
            if not np.isfinite(c0) or c0 < Config.MIN_C0 or c0 > Config.MAX_C0:
                continue
            g_obs, g_N = self.calculator.accelerations(point['v_obs'], v_bar, point['radius'])
            c0_error = np.nan
            if sigma_vobs is not None and sigma_vbar is not None:
                c0_error = self.calculator.c0_error_propagation(
                    point['v_obs'], v_bar, phi, sigma_vobs, sigma_vbar)
            galaxy_c0.append(c0); galaxy_phi.append(phi)
            galaxy_vobs.append(point['v_obs']); galaxy_vbar.append(v_bar)
            galaxy_radii.append(point['radius'])
            galaxy_g_obs.append(g_obs); galaxy_g_N.append(g_N)
            galaxy_c0_errors.append(c0_error)
            valid_points.append({
                'radius': point['radius'], 'v_obs': point['v_obs'], 'v_bar': v_bar,
                'phi': phi, 'c0': c0, 'c0_error': c0_error, 'g_obs': g_obs, 'g_N': g_N
            })
            self.galaxies_with_valid_c0.add(galaxy_name)
            self.all_c0.append(c0); self.all_phi.append(phi)
            self.all_vobs.append(point['v_obs']); self.all_vbar.append(v_bar)
            self.all_radii.append(point['radius'])
            self.all_g_obs.append(g_obs); self.all_g_N.append(g_N)
            self.all_c0_errors.append(c0_error)
        if len(galaxy_c0) >= Config.MIN_POINTS_PER_GALAXY:
            galaxy_c0 = np.array(galaxy_c0); galaxy_phi = np.array(galaxy_phi)
            galaxy_c0_errors = np.array(galaxy_c0_errors)
            self.galaxy_results[galaxy_name] = {
                'n_points': len(galaxy_c0),
                'radii_range': [min(galaxy_radii), max(galaxy_radii)],
                'c0_mean': np.mean(galaxy_c0), 'c0_median': np.median(galaxy_c0),
                'c0_std': np.std(galaxy_c0),
                'c0_error_mean': np.nanmean(galaxy_c0_errors) if np.any(np.isfinite(galaxy_c0_errors)) else np.nan,
                'phi_mean': np.mean(galaxy_phi),
                'data_points': valid_points
            }

    def analyze_all_galaxies(self, all_galaxies):
        print("\n" + "="*80)
        print("GALAXY ANALYSIS (EXACT c₀)")
        print("="*80)
        for name, pts in tqdm(all_galaxies.items(), desc="Analyzing galaxies"):
            self.process_galaxy(name, pts)
        self.all_c0 = np.array(self.all_c0); self.all_phi = np.array(self.all_phi)
        self.all_vobs = np.array(self.all_vobs); self.all_vbar = np.array(self.all_vbar)
        self.all_radii = np.array(self.all_radii)
        self.all_g_obs = np.array(self.all_g_obs); self.all_g_N = np.array(self.all_g_N)
        self.all_c0_errors = np.array(self.all_c0_errors)
        print(f"\n✓ Analysis complete: {len(self.galaxy_results)} galaxies, {len(self.all_c0)} points")
        return self


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

class StatisticalAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.c0 = analyzer.all_c0
        self.phi = analyzer.all_phi
        self.vobs = analyzer.all_vobs
        self.vbar = analyzer.all_vbar
        self.radii = analyzer.all_radii
        self.g_obs = analyzer.all_g_obs
        self.g_N = analyzer.all_g_N
        self.c0_errors = analyzer.all_c0_errors
        self.galaxy_results = analyzer.galaxy_results
        self.n_total = len(self.c0)
        self.results = {}
        np.random.seed(Config.RANDOM_SEED)

    def bootstrap_ci(self, data, n_samples=None):
        if n_samples is None:
            n_samples = Config.BOOTSTRAP_N_SAMPLES
        n = len(data)
        boot_means = [np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(n_samples)]
        boot_means = np.array(boot_means)
        alpha = 1 - Config.BOOTSTRAP_CONFIDENCE
        return {
            'mean': np.mean(boot_means), 'std': np.std(boot_means),
            'ci_low': np.percentile(boot_means, 100*alpha/2),
            'ci_high': np.percentile(boot_means, 100*(1-alpha/2)),
            'contains_predicted': (np.percentile(boot_means, 100*alpha/2) <= Config.C0_PREDICTED <= np.percentile(boot_means, 100*(1-alpha/2)))
        }

    def t_test(self, data):
        t_stat, p_value = stats.ttest_1samp(data, Config.C0_PREDICTED)
        return {'t_statistic': t_stat, 'p_value': p_value}

    def progressive_cuts_analysis(self):
        print("\n" + "="*80)
        print("PROGRESSIVE CUTS ANALYSIS (EXACT c₀)")
        print("="*80)
        results = {}
        for cut_name, low, high in Config.PROGRESSIVE_CUTS:
            mask = (self.c0 >= low) & (self.c0 <= high)
            c0_cut = self.c0[mask]
            if len(c0_cut) < 10: continue
            retention = 100 * len(c0_cut) / self.n_total
            mean, std = np.mean(c0_cut), np.std(c0_cut)
            boot = self.bootstrap_ci(c0_cut)
            ttest = self.t_test(c0_cut)
            results[cut_name] = {'n_points': len(c0_cut), 'retention': retention,
                                 'mean': mean, 'std': std, 'bootstrap': boot, 't_test': ttest}
            print(f"\n{cut_name} [{low}-{high} km/s]: N={len(c0_cut)} ({retention:.1f}%), mean={mean:.1f}±{std:.1f}, CI=[{boot['ci_low']:.2f},{boot['ci_high']:.2f}], p={ttest['p_value']:.6f}")
        return results

    def full_analysis(self):
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS (EXACT c₀)")
        print("="*80)
        self.results['full'] = {
            'n_points': self.n_total, 'mean': np.mean(self.c0), 'median': np.median(self.c0),
            'std': np.std(self.c0), 'bootstrap': self.bootstrap_ci(self.c0), 't_test': self.t_test(self.c0)
        }
        mask_clean = (self.c0 >= 50) & (self.c0 <= 200)
        c0_clean = self.c0[mask_clean]
        self.results['clean'] = {
            'n_points': len(c0_clean), 'mean': np.mean(c0_clean), 'median': np.median(c0_clean),
            'std': np.std(c0_clean), 'bootstrap': self.bootstrap_ci(c0_clean), 't_test': self.t_test(c0_clean)
        }
        print(f"Clean sample (50-200 km/s): N={len(c0_clean)}, mean={np.mean(c0_clean):.1f}±{np.std(c0_clean):.1f}")
        self.results['progressive_cuts'] = self.progressive_cuts_analysis()
        return self.results


# =============================================================================
# ENHANCED SIMULATION COMPARISON
# =============================================================================

class EnhancedSimulationComparator:
    def __init__(self, analyzer, error_estimator):
        self.analyzer = analyzer; self.error_estimator = error_estimator
        self.df = pd.DataFrame({
            'c0_obs': analyzer.all_c0, 'phi': analyzer.all_phi,
            'v_obs': analyzer.all_vobs, 'v_bar': analyzer.all_vbar,
            'radius': analyzer.all_radii, 'c0_error': analyzer.all_c0_errors
        })
        self.df['is_outlier'] = (self.df['c0_obs'] < 50) | (self.df['c0_obs'] > 200)
        self.sigma_vobs_global = error_estimator.global_vobs_error or Config.SIMULATION_DEFAULT_SIGMA_VOBS
        self.sigma_vbar_global = error_estimator.global_vbar_error or Config.SIMULATION_DEFAULT_SIGMA_VBAR
        self.simulation_results = None; self.correlation_results = None
        print(f"\nENHANCED SIMULATION COMPARISON (using σ_vobs={self.sigma_vobs_global:.2f}, σ_vbar={self.sigma_vbar_global:.2f})")

    def estimate_error_correlation(self):
        # Simplified correlation estimate
        return 0.3

    def run_uncorrelated_simulations(self, n_sim=None):
        if n_sim is None: n_sim = Config.SIMULATION_N_SIM
        sim_scatters, sim_means = [], []
        for _ in tqdm(range(n_sim), desc="Uncorrelated sim"):
            vobs_sim = self.df['v_obs'].values + np.random.normal(0, self.sigma_vobs_global, len(self.df))
            vbar_sim = self.df['v_bar'].values + np.random.normal(0, self.sigma_vbar_global, len(self.df))
            vbar_sim = np.maximum(vbar_sim, 0.1)
            with np.errstate(invalid='ignore', divide='ignore'):
                phi_sim = np.log(vobs_sim / vbar_sim)
                c0_sim = (vobs_sim * np.exp(-phi_sim)) / np.sqrt(np.exp(2*phi_sim) - 1)
            valid = (np.isfinite(c0_sim) & (c0_sim > 0) & (c0_sim < 1000) & (phi_sim > 0))
            if np.sum(valid) > 10:
                sim_scatters.append(np.std(c0_sim[valid]))
                sim_means.append(np.mean(c0_sim[valid]))
        sim_scatters = np.array(sim_scatters); sim_means = np.array(sim_means)
        self.simulation_results = {
            'scatters': sim_scatters, 'means': sim_means,
            'mean_scatter': np.mean(sim_scatters), 'std_scatter': np.std(sim_scatters),
            'ci_low': np.percentile(sim_scatters, 2.5), 'ci_high': np.percentile(sim_scatters, 97.5),
            'mean_of_means': np.mean(sim_means), 'std_of_means': np.std(sim_means)
        }
        print(f"✓ Uncorrelated: expected scatter {self.simulation_results['mean_scatter']:.1f}±{self.simulation_results['std_scatter']:.1f} km/s")
        return self.simulation_results

    def run_correlated_simulations(self, n_sim=None, correlation=None):
        if n_sim is None: n_sim = Config.SIMULATION_N_SIM
        if correlation is None: correlation = self.estimate_error_correlation()
        cov = np.array([[self.sigma_vobs_global**2, correlation*self.sigma_vobs_global*self.sigma_vbar_global],
                        [correlation*self.sigma_vobs_global*self.sigma_vbar_global, self.sigma_vbar_global**2]])
        sim_scatters, sim_means = [], []
        for _ in tqdm(range(n_sim), desc="Correlated sim"):
            errors = np.random.multivariate_normal([0,0], cov, len(self.df))
            vobs_sim = self.df['v_obs'].values + errors[:,0]
            vbar_sim = self.df['v_bar'].values + errors[:,1]
            vbar_sim = np.maximum(vbar_sim, 0.1)
            with np.errstate(invalid='ignore', divide='ignore'):
                phi_sim = np.log(vobs_sim / vbar_sim)
                c0_sim = (vobs_sim * np.exp(-phi_sim)) / np.sqrt(np.exp(2*phi_sim) - 1)
            valid = (np.isfinite(c0_sim) & (c0_sim > 0) & (c0_sim < 1000) & (phi_sim > 0))
            if np.sum(valid) > 10:
                sim_scatters.append(np.std(c0_sim[valid]))
                sim_means.append(np.mean(c0_sim[valid]))
        sim_scatters = np.array(sim_scatters); sim_means = np.array(sim_means)
        self.correlation_results = {
            'correlation': correlation,
            'scatters': sim_scatters, 'means': sim_means,
            'mean_scatter': np.mean(sim_scatters), 'std_scatter': np.std(sim_scatters),
            'ci_low': np.percentile(sim_scatters, 2.5), 'ci_high': np.percentile(sim_scatters, 97.5),
            'mean_of_means': np.mean(sim_means), 'std_of_means': np.std(sim_means)
        }
        print(f"✓ Correlated (ρ={correlation}): expected scatter {self.correlation_results['mean_scatter']:.1f}±{self.correlation_results['std_scatter']:.1f} km/s")
        return self.correlation_results

    def plot_enhanced_comparison(self, output_dir):
        if self.simulation_results is None: self.run_uncorrelated_simulations()
        if self.correlation_results is None: self.run_correlated_simulations()
        fig, axes = plt.subplots(2,2,figsize=(14,12))
        observed_scatter = np.std(self.df['c0_obs'][~self.df['is_outlier']])
        axes[0,0].hist(self.simulation_results['scatters'], bins=50, alpha=0.7, color='blue', density=True, label='Uncorrelated')
        axes[0,0].axvline(observed_scatter, color='red', linestyle='--', label=f'Observed: {observed_scatter:.1f}')
        axes[0,0].axvline(self.simulation_results['mean_scatter'], color='green', linestyle=':', label=f'Expected: {self.simulation_results["mean_scatter"]:.1f}')
        axes[0,0].set_title('(a) Uncorrelated Errors'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
        axes[0,1].hist(self.correlation_results['scatters'], bins=50, alpha=0.7, color='purple', density=True, label=f'Correlated (ρ={self.correlation_results["correlation"]:.2f})')
        axes[0,1].axvline(observed_scatter, color='red', linestyle='--', label=f'Observed: {observed_scatter:.1f}')
        axes[0,1].axvline(self.correlation_results['mean_scatter'], color='green', linestyle=':', label=f'Expected: {self.correlation_results["mean_scatter"]:.1f}')
        axes[0,1].set_title('(b) Correlated Errors'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
        categories = ['Observed', 'Uncorrelated\nExpected', 'Correlated\nExpected']
        values = [observed_scatter, self.simulation_results['mean_scatter'], self.correlation_results['mean_scatter']]
        errors = [0, self.simulation_results['std_scatter'], self.correlation_results['std_scatter']]
        axes[1,0].bar(categories, values, yerr=errors, color=['red','blue','purple'], capsize=5, alpha=0.7)
        axes[1,0].set_title('(c) Scatter Comparison'); axes[1,0].grid(True, alpha=0.3, axis='y')
        axes[1,1].axis('off')
        unc_sig = (observed_scatter - self.simulation_results['mean_scatter'])/self.simulation_results['std_scatter']
        cor_sig = (observed_scatter - self.correlation_results['mean_scatter'])/self.correlation_results['std_scatter']
        summary = (f"OBSERVED: {observed_scatter:.1f} km/s\n"
                   f"Uncorr: {self.simulation_results['mean_scatter']:.1f}±{self.simulation_results['std_scatter']:.1f} ({unc_sig:.1f}σ)\n"
                   f"Corr (ρ={self.correlation_results['correlation']:.2f}): {self.correlation_results['mean_scatter']:.1f}±{self.correlation_results['std_scatter']:.1f} ({cor_sig:.1f}σ)")
        axes[1,1].text(0.05, 0.95, summary, transform=axes[1,1].transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.suptitle('Enhanced Error Analysis with Correlated Simulations (Exact c₀)', fontsize=16, y=1.02)
        plt.tight_layout()
        filename = Path(output_dir) / 'figure_S3_enhanced_error_analysis.pdf'
        plt.savefig(filename, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
        return {'uncorrelated': self.simulation_results, 'correlated': self.correlation_results, 'observed_scatter': observed_scatter}


# =============================================================================
# STRATIFIED ANALYSIS
# =============================================================================

class StratifiedAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.df = pd.DataFrame({
            'c0_corrected': analyzer.all_c0, 'phi': analyzer.all_phi,
            'radius_kpc': analyzer.all_radii, 'v_obs': analyzer.all_vobs, 'v_bar': analyzer.all_vbar
        })
        self.df['is_outlier'] = (self.df['c0_corrected'] < 50) | (self.df['c0_corrected'] > 200)
        print(f"\nSTRATIFIED ANALYSIS: {len(self.df)} points")

    def analyze_by_radius(self):
        self.df['radius_bin'] = pd.cut(self.df['radius_kpc'], bins=Config.RADIUS_BINS)
        results = {}
        for bin_name, group in self.df.groupby('radius_bin'):
            if len(group) < 5: continue
            c0_vals = group['c0_corrected']; mean, std = c0_vals.mean(), c0_vals.std()
            scatter_pct = 100*std/mean if mean>0 else 0
            results[str(bin_name)] = {'n_points':len(group),'mean':mean,'std':std,'scatter_percent':scatter_pct,
                                      'outlier_fraction':100*group['is_outlier'].sum()/len(group)}
            print(f"  Radius {bin_name}: N={len(group)}, mean={mean:.1f}±{std:.1f}, scatter={scatter_pct:.1f}%")
        return results

    def analyze_by_phi(self):
        self.df['phi_bin'] = pd.cut(self.df['phi'], bins=Config.PHI_BINS)
        results = {}
        for bin_name, group in self.df.groupby('phi_bin'):
            if len(group) < 5: continue
            c0_vals = group['c0_corrected']; mean, std = c0_vals.mean(), c0_vals.std()
            scatter_pct = 100*std/mean if mean>0 else 0
            results[str(bin_name)] = {'n_points':len(group),'mean':mean,'std':std,'scatter_percent':scatter_pct,
                                      'outlier_fraction':100*group['is_outlier'].sum()/len(group)}
            print(f"  φ {bin_name}: N={len(group)}, mean={mean:.1f}±{std:.1f}, scatter={scatter_pct:.1f}%")
        return results

    def plot_results(self, output_dir):
        fig, axes = plt.subplots(2,2,figsize=(12,10))
        good = self.df[~self.df['is_outlier']]; outliers = self.df[self.df['is_outlier']]
        axes[0,0].scatter(good['radius_kpc'], good['c0_corrected'], alpha=0.3, s=5, color='blue', label='Good')
        axes[0,0].scatter(outliers['radius_kpc'], outliers['c0_corrected'], alpha=0.5, s=10, color='red', label='Outliers')
        axes[0,0].axhline(Config.C0_PREDICTED, color='green', linestyle='--', label=f'c₀={Config.C0_PREDICTED}')
        axes[0,0].set_xscale('log'); axes[0,0].set_xlabel('Radius [kpc]'); axes[0,0].set_ylabel('c₀ [km/s]')
        axes[0,0].set_title('(a) c₀ vs Radius'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
        axes[0,1].scatter(good['phi'], good['c0_corrected'], alpha=0.3, s=5, color='blue')
        axes[0,1].scatter(outliers['phi'], outliers['c0_corrected'], alpha=0.5, s=10, color='red')
        axes[0,1].axhline(Config.C0_PREDICTED, color='green', linestyle='--')
        axes[0,1].set_xlabel('φ'); axes[0,1].set_ylabel('c₀ [km/s]'); axes[0,1].set_title('(b) c₀ vs φ'); axes[0,1].grid(True, alpha=0.3)
        radius_res = self.analyze_by_radius()
        if radius_res:
            bins = list(radius_res.keys()); scatters = [radius_res[b]['scatter_percent'] for b in bins]
            axes[1,0].plot(range(len(bins)), scatters, 'o-', color='purple')
            axes[1,0].set_xticks(range(len(bins))); axes[1,0].set_xticklabels(bins, rotation=45)
            axes[1,0].set_ylabel('Scatter [%]'); axes[1,0].set_title('(c) Scatter vs Radius'); axes[1,0].grid(True, alpha=0.3)
        phi_res = self.analyze_by_phi()
        if phi_res:
            bins = list(phi_res.keys()); scatters = [phi_res[b]['scatter_percent'] for b in bins]
            axes[1,1].plot(range(len(bins)), scatters, 'o-', color='brown')
            axes[1,1].set_xticks(range(len(bins))); axes[1,1].set_xticklabels(bins, rotation=45)
            axes[1,1].set_ylabel('Scatter [%]'); axes[1,1].set_title('(d) Scatter vs φ'); axes[1,1].grid(True, alpha=0.3)
        plt.suptitle('Stratified Analysis of c₀ Scatter (Exact c₀)', fontsize=16)
        plt.tight_layout()
        filename = Path(output_dir) / 'figure_S1_stratified_analysis.pdf'
        plt.savefig(filename, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")
        return {'by_radius': radius_res, 'by_phi': phi_res}


# =============================================================================
# VISUALIZATION (MAIN FIGURES) – revised version
# =============================================================================

def classify_regime(phi_mean):
    """Assign a saturation regime to a galaxy based on its mean φ."""
    if phi_mean < 0.25:
        return 'Non-saturated (Slope)'
    elif phi_mean > 0.55:
        return 'Dwarf (Valley)'
    else:
        return 'Saturated (Plateau)'


class MainVisualizer:
    def __init__(self, analyzer, results, output_dir):
        self.analyzer = analyzer; self.results = results
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({'font.family':'serif','font.size':11,'axes.labelsize':12,'axes.titlesize':14,'legend.fontsize':10,'figure.dpi':Config.FIGURE_DPI})

    def figure_1_c0_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
        full = self.results['full']
        full_means = np.random.normal(full['bootstrap']['mean'], full['bootstrap']['std'], 10000)
        ax1.hist(full_means, bins=50, alpha=0.7, color='steelblue', density=True)
        ax1.axvline(Config.C0_PREDICTED, color='red', linestyle='--', label=f'Predicted {Config.C0_PREDICTED}')
        ax1.axvline(full['bootstrap']['mean'], color='black', linestyle='-', label=f'Mean {full["bootstrap"]["mean"]:.1f}')
        ax1.axvspan(full['bootstrap']['ci_low'], full['bootstrap']['ci_high'], alpha=0.2, color='gray')
        ax1.set_xlabel('c₀ [km/s]'); ax1.set_ylabel('Density'); ax1.set_title(f'(a) Full sample N={full["n_points"]}')
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_xlim(0, 200)
        clean = self.results['clean']
        clean_means = np.random.normal(clean['bootstrap']['mean'], clean['bootstrap']['std'], 10000)
        ax2.hist(clean_means, bins=50, alpha=0.7, color='darkgreen', density=True)
        ax2.axvline(Config.C0_PREDICTED, color='red', linestyle='--', label=f'Predicted {Config.C0_PREDICTED}')
        ax2.axvline(clean['bootstrap']['mean'], color='black', linestyle='-', label=f'Mean {clean["bootstrap"]["mean"]:.1f}')
        ax2.axvspan(clean['bootstrap']['ci_low'], clean['bootstrap']['ci_high'], alpha=0.2, color='gray')
        ax2.set_xlabel('c₀ [km/s]'); ax2.set_ylabel('Density'); ax2.set_title(f'(b) Clean sample (50-200) N={clean["n_points"]}')
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_xlim(0, 200)
        plt.tight_layout()
        filename = self.output_dir / 'figure_1_c0_distribution.pdf'
        plt.savefig(filename, bbox_inches='tight'); plt.close()
        print(f"✓ Saved: {filename}")

    def figure_2_progressive_cuts(self):
        cuts = self.results['progressive_cuts']; names = list(cuts.keys())
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,10))
        x = np.arange(len(names)); means = [cuts[n]['mean'] for n in names]
        ci_low = [cuts[n]['bootstrap']['ci_low'] for n in names]
        ci_high = [cuts[n]['bootstrap']['ci_high'] for n in names]
        ax1.errorbar(x, means, yerr=[[m-l for m,l in zip(means,ci_low)], [h-m for m,h in zip(means,ci_high)]], fmt='o', capsize=5, color='steelblue')
        ax1.axhline(Config.C0_PREDICTED, color='red', linestyle='--', label=f'Predicted {Config.C0_PREDICTED}')
        ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45); ax1.set_ylabel('Mean c₀ [km/s]'); ax1.set_title('(a) Mean c₀ with 95% CI')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        pvals = [cuts[n]['t_test']['p_value'] for n in names]; ret = [cuts[n]['retention'] for n in names]
        ax2.semilogy(x, pvals, 'o-', color='blue', label='p-value')
        ax2.axhline(0.05, color='red', linestyle='--', label='p=0.05')
        ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=45); ax2.set_ylabel('p-value'); ax2.set_title('(b) p-values and retention')
        ax2.legend(loc='upper left')
        ax2t = ax2.twinx(); ax2t.bar(x, ret, alpha=0.3, color='orange'); ax2t.set_ylabel('Retention [%]'); ax2t.set_ylim(0,105)
        plt.tight_layout()
        filename = self.output_dir / 'figure_2_progressive_cuts.pdf'
        plt.savefig(filename, bbox_inches='tight'); plt.close()
        print(f"✓ Saved: {filename}")

    def figure_c0_vs_radius_outliers(self):
        """c₀ vs galactocentric radius with outliers highlighted."""
        df = pd.DataFrame({
            'c0': self.analyzer.all_c0,
            'radius': self.analyzer.all_radii
        })
        mask_low = df['c0'] < Config.C0_LOW
        mask_high = df['c0'] > Config.C0_HIGH
        mask_good = ~(mask_low | mask_high)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df['radius'][mask_good], df['c0'][mask_good],
                   alpha=0.3, s=6, color='steelblue', label=f'Retained (N={mask_good.sum()})')
        ax.scatter(df['radius'][mask_low], df['c0'][mask_low],
                   alpha=0.6, s=12, color='red', marker='o', label=f'c₀ < {Config.C0_LOW} km/s')
        ax.scatter(df['radius'][mask_high], df['c0'][mask_high],
                   alpha=0.6, s=12, color='darkred', marker='s', label=f'c₀ > {Config.C0_HIGH} km/s')
        ax.axhline(Config.C0_LOW, color='red', linestyle='--', alpha=0.7)
        ax.axhline(Config.C0_HIGH, color='red', linestyle='--', alpha=0.7)
        ax.axhline(Config.C0_PREDICTED, color='green', linestyle='-', label=f'c₀ = {Config.C0_PREDICTED}')
        ax.set_xscale('log')
        ax.set_xlabel('Galactocentric Radius [kpc]')
        ax.set_ylabel('c₀ [km/s]')
        ax.set_ylim(0, 350)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        filename = self.output_dir / 'figure_c0_vs_radius_outliers.pdf'
        plt.savefig(filename, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")

    def figure_3_phi_vs_vbar(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use hist2d to avoid black patches
        h = ax.hist2d(self.analyzer.all_vbar, self.analyzer.all_phi,
                      bins=(60, 40), cmap='plasma', norm='log')
        plt.colorbar(h[3], ax=ax, label='Number of points')
        v_range = np.linspace(0, 300, 100)
        ax.plot(v_range, 0.5*np.log(1 + (v_range/Config.C0_PREDICTED)**2),
                'r-', linewidth=3, label='Theoretical')
        ax.set_xlabel(r'$v_{\rm bar}$ [km s$^{-1}$]')
        ax.set_ylabel(r'$\phi$')
        ax.legend()
        ax.grid(True, alpha=0.3)
        filename = self.output_dir / 'figure_3_phi_vs_vbar.pdf'
        plt.savefig(filename, bbox_inches='tight'); plt.close()
        print(f"✓ Saved: {filename}")

    def figure_4_rar_comparison(self):
        """Figure 4: RAR with causal prediction, MOND, and shaded low-acceleration region."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Clean sample points only (50-200 km/s)
        mask_clean = (self.analyzer.all_c0 >= 50) & (self.analyzer.all_c0 <= 200)
        g_N_clean = self.analyzer.all_g_N[mask_clean]
        g_obs_clean = self.analyzer.all_g_obs[mask_clean]

        # Hexbin of full sample in greyscale to show density
        hb = ax.hexbin(self.analyzer.all_g_N, self.analyzer.all_g_obs,
                       gridsize=50, cmap='Greys', mincnt=1, linewidths=0.2, alpha=0.6)
        plt.colorbar(hb, ax=ax, label='Number of points (full sample)')

        # Clean sample points as transparent layer for clarity
        ax.scatter(g_N_clean, g_obs_clean, s=2, alpha=0.2, color='black', label='Clean sample')

        # Unity line
        minv, maxv = 1e-14, 1e-8
        ax.plot([minv, maxv], [minv, maxv], 'k--', label=r'$g_{\rm obs}=g_{\rm bar}$')

        # Causal saturation line
        amp = np.exp(2 * Config.PHI_MAX)
        ax.plot([minv, maxv], [minv, amp * maxv], 'r-', linewidth=2.5,
                label=r'Causal: $g_{\rm obs}=%.3f\,g_{\rm bar}$' % amp)

        # MOND curve
        a0 = 1.2e-10  # m/s²
        g_N_range = np.logspace(-14, -8, 200)
        nu_mond = 0.5 + np.sqrt(0.25 + 1.0 / np.maximum(g_N_range / a0, 1e-12))
        g_obs_mond = g_N_range * nu_mond
        ax.plot(g_N_range, g_obs_mond, 'b--', linewidth=1.8,
                label=r'MOND ($a_0=1.2\times10^{-10}$ m/s²)')

        # Shaded low-acceleration region (future tests)
        ax.axvspan(1e-14, 1e-12, alpha=0.08, color='orange')
        ax.text(3e-13, 2e-12, 'Low-acceleration\nregime\n(future tests)',
                fontsize=8, ha='center', color='darkorange', fontstyle='italic')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(minv, maxv)
        ax.set_ylim(minv, maxv)
        ax.set_xlabel(r'$g_{\rm bar}$ [m s$^{-2}$]')
        ax.set_ylabel(r'$g_{\rm obs}$ [m s$^{-2}$]')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        filename = self.output_dir / 'figure_4_rar_comparison.pdf'
        plt.savefig(filename, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")

    def create_all_figures(self):
        print("\nGENERATING MAIN FIGURES (Exact c₀)")
        self.figure_1_c0_distribution()
        self.figure_2_progressive_cuts()
        self.figure_c0_vs_radius_outliers()   # New figure
        self.figure_3_phi_vs_vbar()
        self.figure_4_rar_comparison()        # Modified figure
        print("✓ All main figures generated.")


# =============================================================================
# REPORT GENERATION
# =============================================================================

class ReportGenerator:
    def __init__(self, analyzer, results, sim_results, output_dir):
        self.analyzer = analyzer; self.results = results; self.sim_results = sim_results
        self.output_dir = Path(output_dir); self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_report(self):
        lines = []
        lines.append("SPARC EXACT c₀ STATISTICAL REPORT")
        lines.append(f"Generated: {datetime.now()}")
        full = self.results['full']; clean = self.results['clean']
        lines.append(f"Full sample: N={full['n_points']}, mean={full['mean']:.2f}±{full['std']:.2f}, CI=[{full['bootstrap']['ci_low']:.2f},{full['bootstrap']['ci_high']:.2f}], p={full['t_test']['p_value']:.6f}")
        lines.append(f"Clean sample: N={clean['n_points']}, mean={clean['mean']:.2f}±{clean['std']:.2f}, CI=[{clean['bootstrap']['ci_low']:.2f},{clean['bootstrap']['ci_high']:.2f}], p={clean['t_test']['p_value']:.6f}")
        filename = self.output_dir / f"statistical_report_{self.timestamp}.txt"
        with open(filename, 'w') as f: f.write('\n'.join(lines))
        print(f"✓ Report saved: {filename}")


# =============================================================================
# DATA EXPORT
# =============================================================================

class DataExporter:
    def __init__(self, analyzer, results, output_dir):
        self.analyzer = analyzer; self.results = results
        self.output_dir = Path(output_dir); self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_all(self):
        df = pd.DataFrame({
            'c0_corrected': self.analyzer.all_c0, 'c0_error': self.analyzer.all_c0_errors,
            'phi': self.analyzer.all_phi, 'v_obs': self.analyzer.all_vobs,
            'v_bar': self.analyzer.all_vbar, 'radius_kpc': self.analyzer.all_radii,
            'g_obs': self.analyzer.all_g_obs, 'g_N': self.analyzer.all_g_N
        })
        meas_file = self.output_dir / f"all_measurements_{self.timestamp}.csv"
        df.to_csv(meas_file, index=False); print(f"✓ Saved: {meas_file}")
        galaxy_rows = []
        for name, data in self.analyzer.galaxy_results.items():
            galaxy_rows.append({
                'galaxy': name, 'n_points': data['n_points'],
                'r_min': data['radii_range'][0], 'r_max': data['radii_range'][1],
                'c0_mean': data['c0_mean'], 'c0_median': data['c0_median'],
                'c0_std': data['c0_std'], 'c0_error_mean': data.get('c0_error_mean', np.nan),
                'phi_mean': data['phi_mean']
            })
        df_gal = pd.DataFrame(galaxy_rows)
        # Add classification column
        df_gal['regime'] = df_gal['phi_mean'].apply(classify_regime)
        gal_file = self.output_dir / f"galaxy_statistics_{self.timestamp}.csv"
        df_gal.to_csv(gal_file, index=False); print(f"✓ Saved: {gal_file}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class SPARCCompleteAnalysisPipeline:
    def __init__(self):
        self.fetcher = SPARCDataFetcher()
        self.error_estimator = ErrorEstimator()
        self.output_dir = Path(Config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n" + "="*100)
        print("SPARC COMPLETE ANALYSIS FOR CAUSAL FIELD MODEL (EXACT c₀)")
        print("="*100)
        try:
            raw_text = self.fetcher.fetch_data()
            galaxies = self.fetcher.parse_raw_data(raw_text)
            sigma_vobs, sigma_vbar = self.error_estimator.estimate_global_errors(galaxies)
            self.analyzer = GalaxyAnalyzer(self.error_estimator)
            self.analyzer.analyze_all_galaxies(galaxies)
            if len(self.analyzer.all_c0) == 0: return False
            stats = StatisticalAnalyzer(self.analyzer)
            self.results = stats.full_analysis()
            stratified = StratifiedAnalyzer(self.analyzer)
            stratified.plot_results(self.output_dir)
            sim = EnhancedSimulationComparator(self.analyzer, self.error_estimator)
            sim.run_uncorrelated_simulations()
            sim.run_correlated_simulations()
            self.sim_results = sim.plot_enhanced_comparison(self.output_dir)
            self.sim_results['sigma_vobs'] = sigma_vobs; self.sim_results['sigma_vbar'] = sigma_vbar
            if MCMC_AVAILABLE:
                # Bayesian omitted for brevity, but could be added
                pass
            visualizer = MainVisualizer(self.analyzer, self.results, self.output_dir)
            visualizer.create_all_figures()
            report = ReportGenerator(self.analyzer, self.results, self.sim_results, self.output_dir)
            report.generate_report()
            exporter = DataExporter(self.analyzer, self.results, self.output_dir)
            exporter.export_all()
            print("\n✓ ANALYSIS COMPLETE")
            print(f"Galaxies with valid c₀: {len(self.analyzer.galaxy_results)}")
            print(f"Total c₀ measurements: {len(self.analyzer.all_c0)}")
            clean = self.results['clean']
            print(f"Clean sample (50-200): {clean['n_points']} points, mean={clean['mean']:.1f}±{clean['std']:.1f}, p={clean['t_test']['p_value']:.6f}")
            return True
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}"); import traceback; traceback.print_exc(); return False


def main():
    pipeline = SPARCCompleteAnalysisPipeline()
    success = pipeline.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())