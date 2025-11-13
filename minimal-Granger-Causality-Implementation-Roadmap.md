# Granger Causality Implementation Roadmap (Minimal)

## Overview

Essential steps only for Granger causality analysis on infant EEG dataset (103 subjects).

---

## Step 1: Load Data
- Load EDF file with MNE
- Extract EEG channels and sampling rate

## Step 2: Preprocess
- Apply bandpass filter (0.5-30 Hz)
- Segment data into 10-second windows

## Step 3: Compute Granger Causality
- Select model order using BIC
- Fit VAR model on each window
- Calculate GC values for all channel pairs
- Average across windows

## Step 4: Save Results
- Save GC matrix as `.npy` file
- Save simple visualization (heatmap)

## Step 5: Repeat for All Subjects
- Loop through all 103 subjects
- Process each session independently

## Step 6: Group Analysis
- Load all GC matrices
- Compute group average
- Create group heatmap

---

## Execution

**Libraries needed**: `mne`, `numpy`, `statsmodels`, `matplotlib`

**Time estimate**: ~2-4 hours for all subjects

**Output**: 
- `results/sub-NORB#####_ses-#_gc_matrix.npy` (one per session)
- `results/group_average_gc.npy`
- Basic heatmap plots

---
