# Granger Causality Implementation Roadmap

## Overview

This roadmap outlines the step-by-step implementation of Granger causality analysis for the infant resting state EEG dataset. The code will process all 103 subjects with varying numbers of sessions (1-4 sessions per subject) and generate comprehensive outputs for analysis.

## Notebook Structure

### Section 1: Configuration and Parameters

**Purpose**: Centralize all tunable parameters in one place for easy adjustment.

**Parameters to Include**:

#### Data Parameters
- Dataset base path
- Output directory structure
- Subject list (manual or automatic detection)
- Session handling strategy

#### Preprocessing Parameters
- High-pass filter cutoff (e.g., 0.5 Hz)
- Low-pass filter cutoff (e.g., 30 Hz)
- Notch filter frequency (60 Hz for power line)
- Resampling rate (if needed, or keep at 200 Hz)
- Reference type (keep common or re-reference)

#### Segmentation Parameters
- Window length in seconds (e.g., 10, 20, 30 seconds)
- Window overlap percentage (e.g., 50%)
- Minimum acceptable segment duration
- Whether to use eyes-closed segments only

#### Granger Causality Parameters
- Model order selection method (AIC, BIC, or fixed value)
- Maximum model order to test (e.g., 50 lags)
- Minimum model order to test (e.g., 1 lag)
- GC computation method (time-domain or frequency-domain)
- Frequency bands for spectral analysis:
  - Delta: 0.5-4 Hz
  - Theta: 4-8 Hz
  - Alpha: 8-13 Hz
  - Beta: 13-30 Hz
  - Gamma: 30-50 Hz (if applicable)

#### Statistical Parameters
- Significance threshold (e.g., p < 0.05)
- Multiple comparison correction method (FDR, Bonferroni, permutation)
- Number of surrogate iterations for permutation testing (e.g., 1000)
- Confidence interval level (e.g., 95%)

#### Analysis Parameters
- Whether to compute pairwise or conditional GC
- Channel pairs to analyze (all pairs, or specific ROIs)
- Whether to compute bidirectional metrics
- Age grouping bins (if doing age-based analysis)

#### Visualization Parameters
- Color maps for connectivity matrices
- Figure DPI and size
- File format for saving plots (PNG, SVG, PDF)
- Whether to generate individual or group-level plots

#### Computational Parameters
- Number of parallel processes
- Memory limits per process
- Verbose output level
- Random seed for reproducibility

---

### Section 2: Environment Setup and Imports

**Purpose**: Load all necessary libraries and verify installation.

**Libraries Needed**:
- **Core**: `numpy`, `pandas`, `scipy`
- **EEG Processing**: `mne`, `pyedflib`
- **Statistical Modeling**: `statsmodels` (VAR models)
- **Granger Causality**: `nitime` or custom implementation
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Network Analysis**: `networkx` (for graph metrics)
- **Parallel Processing**: `joblib` or `multiprocessing`
- **Utilities**: `os`, `json`, `glob`, `warnings`

**Tasks**:
- Import all libraries
- Check library versions
- Set plotting backends
- Configure logging

---

### Section 3: Data Discovery and Inventory

**Purpose**: Scan the dataset and create a comprehensive inventory of all available data.

**Tasks**:
1. Load `participants.tsv` for subject metadata
2. Scan dataset directory structure
3. For each subject:
   - Detect all available sessions
   - Record session count per subject
   - Extract age at acquisition from scans.tsv files
   - Check for existence of:
     - EEG data files (.edf)
     - Channel information
     - Electrode coordinates
     - Event markers
     - Derivative annotations
4. Create a master dataframe with columns:
   - `subject_id`
   - `session_id`
   - `age_months` (converted from decimal years)
   - `sex`
   - `eeg_file_path`
   - `duration_sec`
   - `n_channels`
   - `has_annotations`
   - `file_exists` (validation flag)
   - `session_count_for_subject`
5. Generate summary statistics:
   - Total subjects, total sessions
   - Distribution of session counts
   - Age distribution across sessions
   - Missing data report

**Output**:
- `subject_session_inventory.csv`: Master inventory file
- `data_summary_statistics.txt`: Text report of dataset overview

---

### Section 4: Data Loading and Validation

**Purpose**: Create functions to load and validate individual EEG recordings.

**Functions to Implement**:

1. **`load_eeg_data(subject_id, session_id)`**
   - Load EDF file using MNE or pyedflib
   - Load associated metadata (JSON, TSV files)
   - Load annotations from derivatives
   - Return raw data object with metadata

2. **`validate_eeg_data(raw)`**
   - Check sampling rate consistency
   - Verify channel count and names
   - Check for flat channels
   - Identify bad channels automatically
   - Return validation report

3. **`load_events_and_annotations(subject_id, session_id)`**
   - Load events.tsv (eyes open/closed markers)
   - Load derivative annotations (clean segments)
   - Merge and reconcile different annotation sources
   - Return combined annotations dataframe

**Tasks**:
- Test loading on a sample subject
- Visualize raw data for quality check
- Document any data inconsistencies

**Output**:
- Sample plots showing raw EEG from 2-3 subjects
- Data quality report identifying any issues

---

### Section 5: Preprocessing Pipeline

**Purpose**: Clean and prepare EEG data for Granger causality analysis.

**Functions to Implement**:

1. **`preprocess_eeg(raw, params)`**
   - Apply bandpass filter (high-pass, low-pass)
   - Apply notch filter (60 Hz)
   - Re-reference if needed (common or average reference)
   - Mark bad channels
   - Interpolate bad channels if necessary
   - Return preprocessed raw object

2. **`segment_data(raw, annotations, params)`**
   - Extract eyes-closed segments (if specified)
   - Create overlapping windows of specified length
   - Reject segments with artifacts based on annotations
   - Return list of clean data segments

3. **`check_stationarity(segment)`**
   - Perform Augmented Dickey-Fuller test per channel
   - Return stationarity flags
   - Optional: apply differencing if non-stationary

4. **`extract_clean_epochs(raw, annotations, params)`**
   - Combine segmentation and artifact rejection
   - Ensure minimum segment duration
   - Return epochs object or array

**Tasks**:
- Process a single subject end-to-end
- Visualize before/after preprocessing
- Check power spectral density changes
- Verify stationarity on sample segments

**Output**:
- Before/after preprocessing comparison plots
- PSD plots showing filter effects
- Stationarity test results for sample data

---

### Section 6: Model Order Selection

**Purpose**: Determine optimal lag order for VAR models.

**Functions to Implement**:

1. **`select_model_order(data, max_order, method='aic')`**
   - Fit VAR models for orders 1 to max_order
   - Compute information criteria (AIC, BIC, HQC)
   - Return optimal order and criteria values

2. **`validate_model_order(data, order)`**
   - Fit VAR with selected order
   - Check residual whiteness (Ljung-Box test)
   - Check model stability (eigenvalues)
   - Return validation metrics

3. **`cross_validate_order(segments, max_order)`**
   - Apply order selection to multiple segments
   - Check consistency across segments
   - Return distribution of optimal orders

**Tasks**:
- Run order selection on 5-10 representative subjects
- Compare AIC vs BIC recommendations
- Visualize information criteria curves
- Decide on fixed order or adaptive strategy

**Output**:
- Model order selection plots (IC vs lag)
- Distribution of optimal orders across subjects
- Recommendation for final model order
- Model validation statistics

---

### Section 7: Granger Causality Computation

**Purpose**: Calculate pairwise and spectral Granger causality for all subjects.

**Functions to Implement**:

1. **`compute_pairwise_gc(segment, order)`**
   - Fit unrestricted and restricted VAR models for each channel pair
   - Compute F-statistic and GC index
   - Perform statistical testing
   - Return GC matrix (channels × channels)

2. **`compute_spectral_gc(segment, order, freqs)`**
   - Fit multivariate VAR model
   - Compute frequency-domain transfer functions
   - Calculate spectral GC for each frequency band
   - Return band-specific GC matrices

3. **`compute_conditional_gc(segment, order)`**
   - Fit full multivariate VAR model
   - Compute conditional GC controlling for all other channels
   - Return conditional GC matrix

4. **`aggregate_gc_across_segments(gc_matrices_list)`**
   - Average GC values across segments
   - Compute standard deviation/confidence intervals
   - Weight by segment quality if needed
   - Return mean GC matrix and variance

**Tasks**:
- Implement for single session first
- Test on multiple subjects
- Handle edge cases (insufficient data, singular matrices)
- Optimize for computational efficiency

**Output**:
- Per-segment GC matrices stored as arrays
- Aggregated GC matrix per session
- Computation time benchmarks

---

### Section 8: Statistical Significance Testing

**Purpose**: Determine which connections are statistically significant.

**Functions to Implement**:

1. **`permutation_test_gc(segment, order, n_permutations)`**
   - Generate surrogate data (phase randomization or time-shifted)
   - Compute GC for each surrogate
   - Build null distribution
   - Return significance thresholds

2. **`fdr_correction(p_values, alpha)`**
   - Apply Benjamini-Hochberg FDR correction
   - Return corrected significance mask

3. **`threshold_gc_matrix(gc_matrix, significance_mask)`**
   - Apply significance threshold
   - Return thresholded GC matrix (non-significant set to 0)

4. **`compute_network_metrics(thresholded_gc)`**
   - Calculate graph theory metrics:
     - Node strength (in-degree, out-degree)
     - Clustering coefficient
     - Path length
     - Network density
     - Hub identification
   - Return metrics dictionary

**Tasks**:
- Run permutation testing on subset of subjects
- Compare parametric vs non-parametric testing
- Apply FDR correction
- Calculate network metrics

**Output**:
- Null distributions for sample subjects
- Significance threshold values
- Corrected p-value matrices
- Binary adjacency matrices (significant connections)

---

### Section 9: Batch Processing Pipeline

**Purpose**: Process all 103 subjects with all their sessions in an automated, robust manner.

**Functions to Implement**:

1. **`process_single_session(subject_id, session_id, params)`**
   - Load data
   - Preprocess
   - Segment
   - Select model order
   - Compute GC
   - Apply significance testing
   - Save results
   - Handle errors gracefully
   - Return success/failure status and results summary

2. **`process_subject_all_sessions(subject_id, params)`**
   - Detect all sessions for subject
   - Process each session independently
   - Store session-specific results
   - Compute within-subject session comparisons (if multiple sessions)
   - Return subject-level summary

3. **`batch_process_all_subjects(subject_list, params, n_jobs)`**
   - Parallel processing across subjects
   - Progress tracking
   - Error logging
   - Resource management
   - Save intermediate results
   - Return batch processing report

**Error Handling Strategy**:
- Try-except blocks for each subject
- Log errors to file with traceback
- Continue processing remaining subjects
- Generate error summary report

**Tasks**:
- Test on small subset (5-10 subjects)
- Validate outputs
- Run full batch processing
- Monitor resource usage

**Output Files per Session**:
- `sub-NORB#####_ses-#_gc_matrix.npy`: Raw GC values
- `sub-NORB#####_ses-#_gc_pvalues.npy`: P-values
- `sub-NORB#####_ses-#_gc_significant.npy`: Binary significance mask
- `sub-NORB#####_ses-#_spectral_gc.npz`: Frequency-band GC (if computed)
- `sub-NORB#####_ses-#_network_metrics.json`: Graph metrics
- `sub-NORB#####_ses-#_processing_log.txt`: Processing details and parameters

**Output Files per Subject**:
- `sub-NORB#####_all_sessions_summary.json`: Metadata for all sessions
- `sub-NORB#####_longitudinal_comparison.csv`: Session comparison (if applicable)

**Output Files for Entire Dataset**:
- `processing_report.txt`: Summary of successful/failed processing
- `error_log.txt`: Detailed error messages
- `computation_stats.csv`: Processing time and resource usage per subject

---

### Section 10: Quality Control and Validation

**Purpose**: Verify the quality and validity of computed Granger causality results.

**Functions to Implement**:

1. **`validate_gc_results(gc_matrix)`**
   - Check for NaN or Inf values
   - Verify diagonal is zero (no self-causation)
   - Check value ranges are reasonable
   - Return validation report

2. **`check_model_assumptions(residuals)`**
   - Test residual whiteness (autocorrelation)
   - Test residual normality
   - Check for heteroskedasticity
   - Return diagnostic statistics

3. **`compare_with_literature(results, expected_patterns)`**
   - Check for expected developmental patterns
   - Verify age-related trends
   - Flag unexpected results for review
   - Return comparison report

**Tasks**:
- Review results from 10-20 random subjects
- Check for consistent patterns
- Identify outliers or anomalies
- Validate against known findings from literature

**Output**:
- Quality control report with pass/fail flags
- Outlier subjects list for manual review
- Validation statistics summary

---

### Section 11: Group-Level Analysis

**Purpose**: Aggregate results across subjects and sessions for population-level insights.

**Functions to Implement**:

1. **`aggregate_gc_across_subjects(results_dict)`**
   - Load all individual GC matrices
   - Compute group mean and standard deviation
   - Handle different session counts per subject appropriately
   - Return group-level statistics

2. **`age_stratified_analysis(results_dict, age_bins)`**
   - Group subjects by age ranges
   - Compute mean GC per age group
   - Test for age-related differences
   - Return age-stratified results

3. **`longitudinal_analysis(multi_session_subjects)`**
   - Extract subjects with multiple sessions
   - Compare GC matrices across sessions within subjects
   - Test for developmental changes
   - Return longitudinal statistics

4. **`sex_comparison(results_dict, sex_labels)`**
   - Separate male and female subjects
   - Compare GC patterns
   - Statistical testing for sex differences
   - Return sex-stratified results

5. **`frequency_band_analysis(spectral_gc_results)`**
   - Aggregate spectral GC across subjects
   - Compare connectivity in different frequency bands
   - Identify band-specific networks
   - Return band-specific statistics

**Tasks**:
- Compute grand average GC matrix
- Perform age correlation analysis
- Analyze longitudinal subjects separately
- Compare frequency bands

**Output**:
- `group_mean_gc_matrix.npy`: Average across all subjects/sessions
- `group_std_gc_matrix.npy`: Standard deviation
- `age_stratified_results.npz`: Results by age group
- `longitudinal_subjects_analysis.csv`: Within-subject changes
- `sex_comparison_results.csv`: Male vs female comparison
- `frequency_band_statistics.csv`: Band-specific connectivity

---

### Section 12: Network Analysis

**Purpose**: Characterize the connectivity network structure using graph theory.

**Functions to Implement**:

1. **`construct_graph(gc_matrix, threshold)`**
   - Convert GC matrix to directed graph
   - Apply threshold to create binary network
   - Return NetworkX graph object

2. **`compute_node_metrics(graph)`**
   - In-degree, out-degree, total degree
   - Betweenness centrality
   - Closeness centrality
   - Eigenvector centrality
   - Return node metrics dataframe

3. **`compute_network_metrics(graph)`**
   - Global efficiency
   - Clustering coefficient
   - Average path length
   - Modularity
   - Small-worldness
   - Return network-level metrics

4. **`identify_hubs(node_metrics, percentile)`**
   - Identify highly connected nodes
   - Classify as source hubs, sink hubs, or relay hubs
   - Return hub classification

5. **`community_detection(graph)`**
   - Apply Louvain or other community detection algorithm
   - Identify functional modules
   - Return community assignments

**Tasks**:
- Apply network analysis to group-average GC
- Analyze individual subject networks
- Compare network properties across age groups
- Identify developmental changes in network topology

**Output**:
- `network_metrics_per_subject.csv`: Graph metrics for each subject
- `hub_regions_group_level.csv`: Identified hub electrodes
- `community_structure.json`: Module assignments
- `network_properties_summary.txt`: Network characteristics

---

### Section 13: Statistical Analysis

**Purpose**: Test hypotheses about developmental trends and group differences.

**Functions to Implement**:

1. **`correlation_with_age(gc_values, ages)`**
   - Compute correlation between GC strength and age
   - Test significance with multiple comparison correction
   - Return correlation coefficients and p-values

2. **`linear_mixed_effects_model(data)`**
   - Model GC ~ age + sex + (1|subject) for longitudinal data
   - Account for repeated measures
   - Return model fit and statistics

3. **`permutation_test_groups(group1, group2, n_permutations)`**
   - Non-parametric test for group differences
   - Return p-values for each connection

4. **`effect_size_calculation(group1, group2)`**
   - Compute Cohen's d for group differences
   - Return effect size matrix

**Tasks**:
- Test age effects on connectivity
- Compare first vs. later sessions in longitudinal subjects
- Test sex differences
- Identify developmentally-changing connections

**Output**:
- `age_correlation_results.csv`: Correlations with significance
- `group_comparison_statistics.csv`: Statistical test results
- `effect_sizes.csv`: Effect sizes for significant differences
- `developmental_trajectories.csv`: Age-related changes

---

### Section 14: Visualization - Individual Level

**Purpose**: Create visualizations for individual subjects and sessions.

**Plots to Generate**:

1. **Connectivity Matrix Heatmap**
   - GC values as color-coded matrix
   - Channels on both axes
   - Separate plots for raw and thresholded matrices
   - Colorbar with appropriate scale

2. **Network Graph**
   - Nodes positioned based on electrode locations
   - Edges represent significant GC
   - Edge thickness proportional to GC strength
   - Separate in/out connections or combined

3. **Circular Connectivity Plot**
   - Channels arranged in circle
   - Curved edges showing connections
   - Color-coded by GC strength

4. **Topographic Maps**
   - Head topography showing outflow/inflow per channel
   - Interpolated scalp maps

5. **Spectral GC Plots**
   - Frequency-resolved GC for selected channel pairs
   - Separate panels for each frequency band

**Functions to Implement**:
- `plot_gc_matrix(gc_matrix, channel_names, title, save_path)`
- `plot_network_graph(gc_matrix, electrode_positions, threshold, save_path)`
- `plot_circular_connectivity(gc_matrix, channel_names, save_path)`
- `plot_topographic_flow(gc_matrix, montage, save_path)`
- `plot_spectral_gc(spectral_gc, freq_bands, save_path)`

**Output per Session**:
- `sub-NORB#####_ses-#_gc_matrix.png`
- `sub-NORB#####_ses-#_network_graph.png`
- `sub-NORB#####_ses-#_circular_plot.png`
- `sub-NORB#####_ses-#_topo_outflow.png`
- `sub-NORB#####_ses-#_topo_inflow.png`
- `sub-NORB#####_ses-#_spectral_gc.png`

---

### Section 15: Visualization - Group Level

**Purpose**: Create visualizations summarizing results across all subjects.

**Plots to Generate**:

1. **Group Average GC Matrix**
   - Mean GC across all subjects
   - Standard deviation matrix
   - Consistency map (% of subjects showing connection)

2. **Age-Related Changes**
   - Scatter plots: GC strength vs. age for top connections
   - Regression lines with confidence intervals
   - Heatmap of age correlation coefficients

3. **Frequency Band Comparison**
   - Side-by-side GC matrices for each band
   - Bar plots comparing connectivity strength across bands
   - Network graphs colored by dominant frequency

4. **Longitudinal Trajectories**
   - Line plots showing within-subject changes over sessions
   - Individual subject lines with group mean overlay
   - Separate plots for different connection types

5. **Network Metrics Distributions**
   - Histograms of node degree distributions
   - Boxplots comparing metrics across age groups
   - Scatter plots of metrics vs. age

6. **Hub Analysis**
   - Brain topography highlighting hub regions
   - Bar charts of hub strength
   - Comparison of hubs across age groups

7. **Statistical Results**
   - Volcano plots (effect size vs. significance)
   - FDR-corrected significance maps
   - Effect size matrices

**Functions to Implement**:
- `plot_group_average_gc(mean_gc, std_gc, save_path)`
- `plot_age_correlations(correlation_matrix, p_values, save_path)`
- `plot_frequency_comparison(band_gc_dict, save_path)`
- `plot_longitudinal_trajectories(multi_session_data, save_path)`
- `plot_network_metrics_summary(metrics_df, save_path)`
- `plot_hub_analysis(hub_data, electrode_positions, save_path)`

**Output**:
- `group_average_gc_matrix.png`
- `group_std_gc_matrix.png`
- `consistency_map.png`
- `age_correlation_heatmap.png`
- `top_connections_vs_age.png`
- `frequency_bands_comparison.png`
- `longitudinal_trajectories.png`
- `network_metrics_distributions.png`
- `hub_regions_topography.png`
- `statistical_results_volcano.png`

---

### Section 16: Interactive Visualizations

**Purpose**: Create interactive plots for exploratory analysis.

**Plots to Generate**:

1. **Interactive Connectivity Matrix** (Plotly)
   - Hover to see exact GC values
   - Click to highlight row/column
   - Toggle between raw and thresholded
   - Zoom and pan capabilities

2. **Interactive Network Graph** (Plotly/NetworkX)
   - 3D brain network visualization
   - Interactive node selection
   - Edge filtering by threshold
   - Animation over age or sessions

3. **Interactive Dashboard** (Plotly Dash or similar)
   - Subject selector dropdown
   - Session selector
   - Multiple coordinated views
   - Parameter adjustment sliders

**Functions to Implement**:
- `create_interactive_matrix(gc_matrix, channel_names)`
- `create_interactive_network(gc_matrix, positions)`
- `create_dashboard_app(all_results)`

**Output**:
- `interactive_gc_matrix.html`
- `interactive_network_3d.html`
- `dashboard_app.py` (if creating separate dashboard)

---

### Section 17: Results Export and Documentation

**Purpose**: Organize and document all outputs for analysis and publication.

**Tasks**:

1. **Create Results Directory Structure**
   ```
   results/
   ├── individual_subjects/
   │   ├── sub-NORB00001/
   │   │   ├── ses-1/
   │   │   │   ├── matrices/
   │   │   │   ├── plots/
   │   │   │   └── metrics/
   │   │   └── summary/
   │   └── ...
   ├── group_level/
   │   ├── matrices/
   │   ├── statistics/
   │   ├── plots/
   │   └── network_analysis/
   ├── quality_control/
   ├── logs/
   └── documentation/
   ```

2. **Generate Summary Reports**
   - Processing summary (subjects completed, failed, time taken)
   - Data quality report
   - Statistical results summary
   - Key findings document

3. **Export Data Tables**
   - CSV files for easy import to R or other tools
   - Excel workbooks with multiple sheets
   - JSON files for web applications

4. **Create Methods Documentation**
   - Document exact parameters used
   - Software versions
   - Processing timestamps
   - Reproducibility information

**Output**:
- `processing_summary_report.pdf`
- `methods_documentation.txt`
- `key_findings_summary.md`
- `results_inventory.csv` (list of all output files)
- `reproducibility_info.json` (parameters, versions, seeds)

---

### Section 18: Validation and Sanity Checks

**Purpose**: Verify results make sense and are scientifically valid.

**Checks to Perform**:

1. **Biological Plausibility**
   - Compare dominant connections with known anatomy
   - Check for expected posterior-anterior patterns
   - Verify frequency-specific connectivity patterns
   - Compare with published infant EEG connectivity studies

2. **Statistical Sanity**
   - Verify p-value distributions (should be uniform under null)
   - Check effect sizes are reasonable
   - Ensure multiple comparison correction is working
   - Validate age correlations with expected developmental trends

3. **Technical Validation**
   - Confirm no data leakage between train/test
   - Verify model assumptions are met
   - Check for overfitting (if using cross-validation)
   - Ensure consistent results across random seeds

4. **Reproducibility Test**
   - Re-run analysis on subset with different random seed
   - Verify results are stable
   - Check sensitivity to parameter changes

**Output**:
- `validation_checklist.txt`: Pass/fail for each check
- `sensitivity_analysis.csv`: Results under different parameters
- `reproducibility_report.txt`: Consistency across runs

---

### Section 19: Interpretation and Reporting

**Purpose**: Synthesize results into meaningful scientific insights.

**Sections to Include**:

1. **Executive Summary**
   - Key findings in 3-5 bullet points
   - Main connectivity patterns observed
   - Developmental trends identified
   - Novel or unexpected results

2. **Descriptive Statistics**
   - Overall connectivity strength
   - Distribution of significant connections
   - Network density across subjects
   - Frequency band preferences

3. **Developmental Findings**
   - Age-related increases/decreases in connectivity
   - Longitudinal changes in multi-session subjects
   - Critical periods or transitions identified
   - Comparison with literature expectations

4. **Network Characterization**
   - Hub regions identified
   - Modular structure
   - Small-world properties
   - Directed flow patterns (anterior-posterior, hemispheric)

5. **Frequency-Specific Results**
   - Dominant bands for connectivity
   - Band-specific network structures
   - Age effects per frequency band

6. **Sex Differences** (if significant)
   - Connections showing sex dimorphism
   - Interaction with age

7. **Limitations and Caveats**
   - Assumptions made
   - Data quality issues
   - Interpretation constraints
   - Future directions

**Output**:
- `analysis_report.md`: Comprehensive markdown report
- `key_results_slides.pptx`: Presentation-ready summary
- `supplementary_tables.xlsx`: Detailed statistics

---

### Section 20: Future Extensions

**Purpose**: Document potential next steps and advanced analyses.

**Possible Extensions**:

1. **Source-Space Analysis**
   - Reconstruct cortical sources
   - Compute GC in source space
   - Reduce volume conduction effects

2. **Time-Varying Connectivity**
   - Sliding window analysis
   - Identify dynamic connectivity states
   - State transition analysis

3. **Graph Theory Extensions**
   - Rich-club analysis
   - Core-periphery structure
   - Motif analysis

4. **Machine Learning Applications**
   - Predict age from connectivity patterns
   - Classification of developmental stages
   - Anomaly detection

5. **Integration with Other Data**
   - Combine with behavioral measures
   - Correlate with structural MRI (if available)
   - Link to clinical outcomes

6. **Advanced Statistical Models**
   - Hierarchical Bayesian models
   - Network-based statistics
   - Multivariate pattern analysis

**Output**:
- `future_directions.md`: Detailed roadmap for extensions

---

## Summary of Expected Outputs

### Data Files
- Subject-session inventory CSV
- Individual GC matrices (NPY format)
- Spectral GC results (NPZ format)
- Network metrics (JSON/CSV)
- Group-level statistics (CSV/NPZ)
- Processing logs (TXT)

### Visualizations
- Individual subject plots (PNG/PDF): ~600-1200 files
- Group-level plots (PNG/PDF): ~20-50 files
- Interactive visualizations (HTML): ~5-10 files

### Statistical Results
- Age correlation matrices
- Group comparison results
- Effect size matrices
- Significance maps

### Documentation
- Processing summary reports
- Methods documentation
- Key findings summary
- Validation reports

### Total Storage Estimate
- Raw results: ~5-10 GB
- Visualizations: ~1-2 GB
- Documentation: ~100 MB
- **Total**: ~6-12 GB

---

## Recommended Execution Order

1. **Test Phase** (Sections 1-10): Run on 5-10 subjects to validate pipeline
2. **Optimization Phase**: Tune parameters, optimize code for speed
3. **Full Batch Phase** (Section 9): Process all subjects
4. **Quality Control** (Section 10): Validate results
5. **Analysis Phase** (Sections 11-13): Group-level statistics
6. **Visualization Phase** (Sections 14-16): Generate all plots
7. **Reporting Phase** (Sections 17-19): Document and interpret

---

## Time Estimates

- Setup and testing: 1-2 days
- Full batch processing: 4-8 hours (depending on parallelization)
- Quality control and validation: 2-4 hours
- Group-level analysis: 4-6 hours
- Visualization generation: 2-4 hours
- Report writing: Variable

**Total**: ~3-5 days of computation and analysis

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Project**: DPCN-Project - Infant EEG Granger Causality Analysis
