# Granger Causality Implementation Roadmap

## Overview

This roadmap outlines a streamlined implementation of Granger causality analysis for the infant EEG dataset (103 subjects, 1-4 sessions each).

---

## Phase 1: Setup & Data Discovery

### 1. Configuration
Set key parameters in one place:
- **Paths**: Dataset location, output directory
- **Preprocessing**: Filters (0.5-30 Hz bandpass, 60 Hz notch)
- **Segmentation**: Window length (e.g., 10s), overlap (50%)
- **GC Parameters**: Model order method (AIC/BIC), max order (~50)
- **Frequency Bands**: Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30) Hz
- **Statistics**: p-value threshold (0.05), FDR correction

### 2. Environment Setup
Import libraries:
- `numpy`, `pandas`, `scipy`, `mne`, `statsmodels`, `matplotlib`, `seaborn`, `networkx`

### 3. Data Inventory
- Scan dataset for all subjects/sessions
- Extract metadata (age, sex, session count)
- Create `subject_session_inventory.csv`
- Generate summary statistics

---

## Phase 2: Core Pipeline (Test on 5-10 Subjects First)

### 4. Load & Validate
- Load EDF files with MNE
- Check data quality (flat channels, sampling rate)
- Load annotations (eyes open/closed, artifacts)

### 5. Preprocessing
- Apply bandpass and notch filters
- Re-reference if needed
- Segment into clean windows (eyes-closed, artifact-free)
- Check stationarity (ADF test)

### 6. Model Order Selection
- Fit VAR models for different orders
- Use AIC/BIC to select optimal order
- Validate with residual tests (whiteness, stability)

### 7. Compute Granger Causality
- For each segment:
  - Fit VAR models (pairwise or multivariate)
  - Calculate GC values for all channel pairs
  - Compute spectral GC for frequency bands (optional)
- Average GC across segments per session

### 8. Statistical Testing
- Permutation testing for significance thresholds
- Apply FDR correction
- Threshold GC matrices (keep only significant connections)

---

## Phase 3: Batch Processing

### 9. Process All Subjects
- Wrap steps 4-8 into `process_single_session()` function
- Add error handling (try-except, logging)
- Run in parallel across subjects
- Save per-session outputs:
  - `sub-NORB#####_ses-#_gc_matrix.npy`
  - `sub-NORB#####_ses-#_gc_significant.npy`
  - `sub-NORB#####_ses-#_processing_log.txt`

### 10. Quality Control
- Check for NaN/Inf values
- Verify GC matrices look reasonable
- Validate against expected patterns
- Generate QC report

---

## Phase 4: Group Analysis & Visualization

### 11. Group-Level Analysis
- Aggregate GC matrices across all subjects
- Compute group mean and std
- Age stratification (correlate GC with age)
- Longitudinal analysis (subjects with multiple sessions)
- Compare frequency bands

### 12. Network Analysis
- Convert GC matrices to directed graphs
- Calculate metrics: degree, clustering, path length, hubs
- Identify hub regions and communities
- Compare network properties across age groups

### 13. Statistical Testing
- Test age correlations (with FDR correction)
- Compare groups (e.g., sex differences, age bins)
- Calculate effect sizes

### 14. Visualizations

**Individual Level:**
- Connectivity matrix heatmaps
- Network graphs with electrode positions
- Topographic maps (inflow/outflow)

**Group Level:**
- Group average GC matrix
- Age correlation plots
- Frequency band comparisons
- Network metrics distributions
- Hub identification plots

---

## Phase 5: Results & Documentation

### 15. Export Results
Organize outputs:
```
results/
├── individual/        # Per-subject matrices and plots
├── group/             # Group statistics and visualizations  
├── quality_control/   # QC reports
└── logs/              # Processing logs
```

### 16. Generate Reports
- Processing summary (successes/failures, timing)
- Statistical results summary
- Key findings document
- Methods documentation for reproducibility

---

## Simplified Execution Plan

1. **Test Phase** (~1-2 days): Run pipeline on 5-10 subjects, validate outputs
2. **Full Processing** (~4-8 hours): Batch process all 103 subjects in parallel
3. **Analysis** (~1 day): Group statistics, network analysis, age effects
4. **Visualization** (~0.5 day): Generate all plots
5. **Documentation** (~0.5 day): Write up findings

**Total Time**: ~3-4 days

---

## Key Outputs

- **Per Session**: GC matrices (.npy), significance masks, plots
- **Group Level**: Mean GC matrix, age correlations, network metrics
- **Documentation**: Processing report, QC report, findings summary

**Storage**: ~6-12 GB total

---

**Document Version**: 2.0 (Simplified)  
**Last Updated**: October 2025  
**Project**: DPCN-Project - Infant EEG Granger Causality Analysis
