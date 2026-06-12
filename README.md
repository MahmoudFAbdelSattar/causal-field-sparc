```markdown
# Causal Field Analysis of SPARC Galaxies (v12.1)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the complete analysis code and data products for the paper:

**"Observational Evidence for a Characteristic Kinematic Speed \(c_0 \approx 104\) km s⁻¹ in Saturated Spiral Galaxies from 1222 Independent Measurements across 115 SPARC Galaxies"**

by Mahmoud F. Abdel-Sattar (2026)

The code implements the Causal Field Model analysis of rotation curves from the SPARC (Spitzer Photometry and Accurate Rotation Curves) database, demonstrating the existence of a universal characteristic speed \(c_0 \approx 104.3\) km s⁻¹ in the saturated regime of spiral galaxies.

## Key Results

- **Sample:** 1222 independent measurements from 115 SPARC galaxies
- **Characteristic speed:** \(c_0 = 104.3\) km s⁻¹
- **Saturation field:** \(\phi_{\max} = 0.429 \pm 0.019\)
- **Gravitational amplification:** \(e^{2\phi} = 2.357\)
- **Clean sample (723 points):** \(\langle c_0 \rangle = 106.9 \pm 43.2\) km s⁻¹
- **Bootstrap 95% CI:** \([103.71, 110.00]\) km s⁻¹
- **t-test p-value:** \(p = 0.112\)
- **Progressive cuts convergence:** mean = 104.3 km s⁻¹ with \(p > 0.98\)

## Repository Structure

```

causal-field-sparc/
├── README.md
├── LICENSE
├── requirements.txt
├── sparc_causal_field_exact_c0_v12_1.py   # Main analysis pipeline (v12.1)
├── all_measurements_YYYYMMDD_HHMMSS.csv   # Complete measurement data
├── galaxy_statistics_YYYYMMDD_HHMMSS.csv  # Galaxy statistics (with regime classification)
└── sparc_complete_analysis_exact_c0/      # Output directory (figures, reports)
├── figure_1_c0_distribution.pdf
├── figure_2_progressive_cuts.pdf
├── figure_c0_vs_radius_outliers.pdf
├── figure_3_phi_vs_vbar.pdf
├── figure_4_rar_comparison.pdf
├── figure_S1_stratified_analysis.pdf
├── figure_S3_enhanced_error_analysis.pdf
└── statistical_report_YYYYMMDD_HHMMSS.txt

```

## Main Analysis Code

`sparc_causal_field_exact_c0_v12_1.py` performs:

1. **Data Acquisition:** Downloads the SPARC database
2. **Error Estimation:** Measures observational uncertainties directly from the data using the Median Absolute Deviation (MAD)
3. **Galaxy Analysis:** Computes \(c_0\) for all valid measurements using the exact definition \(c_0 = v_{\text{bar}} / \sqrt{e^{2\phi} - 1}\)
4. **Statistical Analysis:**
   - Bootstrap confidence intervals (10,000 resamplings)
   - Progressive cuts analysis across five nested quality windows
   - One-sample t-tests against the predicted value
5. **Monte Carlo Simulations:** 10,000 realisations with uncorrelated and correlated errors
6. **Stratified Analysis:** Scatter analysis by galactocentric radius and \(\phi\)
7. **Visualization:** Generates publication-quality figures:
   - Figure 1: Bootstrap distribution of \(c_0\)
   - Figure 2: Progressive cuts convergence
   - Figure 3: \(c_0\) vs. radius (outlier justification)
   - Figure 4: \(\phi\) vs. \(v_{\text{bar}}\)
   - Figure 5: Radial Acceleration Relation (causal model, MOND, and low-acceleration regime)

## New Features in v12.1

- **Exact \(c_0\) definition** used throughout all calculations
- **Saturation regime classification** for all galaxies (Slope, Plateau, Valley)
- **Enhanced RAR figure** with MOND prediction and future test regime
- **Physical justification of quality cuts** via radius-dependent outlier analysis
- **Improved visualization** (hist2d for density, transparent scatter overlays)
- **All comments translated to English** for broader accessibility

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/MahmoudFAbdelSattar/causal-field-sparc.git
   cd causal-field-sparc
```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python sparc_causal_field_exact_c0_v12_1.py
   ```

Dependencies

The main dependencies are:

· numpy
· scipy
· pandas
· matplotlib
· requests
· tqdm
· pymc (optional, for Bayesian analysis)
· arviz (optional, for Bayesian analysis)

See requirements.txt for the complete list with version specifications.

Data Source

The SPARC database is publicly available at:
https://astroweb.cwru.edu/SPARC

Reference: Lelli, F., McGaugh, S.S., & Schombert, J.M. (2016), The Astronomical Journal, 152, 157.

Citation

If you use this code or data in your research, please cite:

```bibtex
@article{AbdelSattar2026,
  title   = {Observational Evidence for a Characteristic Kinematic Speed 
             c₀ ≈ 104 km s⁻¹ in Saturated Spiral Galaxies},
  author  = {Abdel-Sattar, Mahmoud F.},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year    = {2026},
  note    = {Submitted}
}
```

License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

Mahmoud F. Abdel-Sattar

· Department of Astronomy and Meteorology
· Faculty of Science, Al-Azhar University
· Cairo, Egypt
· Email: m.f.abdel-sattar@azhar.edu.eg
· GitHub: MahmoudFAbdelSattar

```
