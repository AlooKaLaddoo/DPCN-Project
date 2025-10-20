# Granger Causality Analysis for EEG Data

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Mathematical Framework](#mathematical-framework)
4. [Parameters and Their Effects](#parameters-and-their-effects)
5. [Application to Infant Resting State EEG](#application-to-infant-resting-state-eeg)
6. [Practical Considerations](#practical-considerations)
7. [References](#references)

## Introduction

Granger causality is a statistical concept used to determine whether one time series can predict another time series. Originally developed by Nobel laureate Clive Granger for econometrics, it has become a fundamental tool in neuroscience for analyzing directional interactions between brain regions.

**Key Concept**: If past values of time series X provide statistically significant information about future values of time series Y beyond what is available from past values of Y alone, then X is said to "Granger-cause" Y.

**Important Note**: Granger causality measures predictive causality, not true causality in the philosophical sense. It identifies statistical precedence and information flow rather than direct causal mechanisms.

## Theoretical Foundation

### Concept of Predictive Causality

Granger causality is based on two fundamental principles:

1. **Temporal Precedence**: The cause occurs before the effect
2. **Predictive Information**: The cause contains unique information about the future of the effect

### Types of Granger Causality

#### Bivariate Granger Causality
Examines the relationship between two time series in isolation. While computationally simple, it can be misleading due to confounding influences from other variables.

#### Conditional Granger Causality
Tests causality between two variables while conditioning on additional variables, reducing spurious causality detection.

#### Multivariate Granger Causality
Simultaneously considers all relevant time series, providing a more accurate picture of the causal network structure.

### Directional Connectivity

Granger causality provides directional connectivity measures:
- **X → Y**: X Granger-causes Y
- **Y → X**: Y Granger-causes X
- **X ↔ Y**: Bidirectional causality
- **X ⊥ Y**: No causal relationship

## Mathematical Framework

### Univariate Autoregressive Model

For a single time series Y, an autoregressive model of order p is:

$$Y_t = \sum_{j=1}^{p} a_j Y_{t-j} + \epsilon_t$$

where:
- $Y_t$ is the value at time t
- $a_j$ are the autoregressive coefficients
- $p$ is the model order (lag)
- $\epsilon_t$ is white noise with variance $\sigma^2$

### Bivariate Autoregressive Model

For two time series X and Y, the bivariate model is:

**Unrestricted Model** (includes past of both X and Y):

$$Y_t = \sum_{j=1}^{p} a_j Y_{t-j} + \sum_{j=1}^{p} b_j X_{t-j} + \epsilon_t$$

**Restricted Model** (includes only past of Y):

$$Y_t = \sum_{j=1}^{p} a_j Y_{t-j} + \eta_t$$

where:
- $a_j$, $b_j$ are regression coefficients
- $\epsilon_t$ has variance $\sigma_{\epsilon}^2$ (unrestricted)
- $\eta_t$ has variance $\sigma_{\eta}^2$ (restricted)

### Granger Causality Measure

X Granger-causes Y if the unrestricted model provides significantly better prediction than the restricted model. The Granger causality index is:

$$F_{X \rightarrow Y} = \ln\left(\frac{\sigma_{\eta}^2}{\sigma_{\epsilon}^2}\right)$$

or equivalently:

$$F_{X \rightarrow Y} = \ln\left(\frac{\text{Var}(\eta)}{\text{Var}(\epsilon)}\right)$$

**Interpretation**:
- $F_{X \rightarrow Y} > 0$: X provides predictive information about Y
- $F_{X \rightarrow Y} = 0$: X does not Granger-cause Y
- Larger values indicate stronger causal influence

### Statistical Testing

The null hypothesis $H_0$: X does not Granger-cause Y is tested using:

$$F = \frac{(RSS_r - RSS_u)/p}{RSS_u/(n-2p-1)}$$

where:
- $RSS_r$ = Residual Sum of Squares (restricted model)
- $RSS_u$ = Residual Sum of Squares (unrestricted model)
- $p$ = number of lags
- $n$ = number of observations

This follows an F-distribution with $(p, n-2p-1)$ degrees of freedom.

### Multivariate Extension (Vector Autoregression)

For a multivariate system with k time series:

$$\mathbf{Y}_t = \sum_{j=1}^{p} \mathbf{A}_j \mathbf{Y}_{t-j} + \mathbf{E}_t$$

where:
- $\mathbf{Y}_t$ is a k-dimensional vector
- $\mathbf{A}_j$ are k×k coefficient matrices
- $\mathbf{E}_t$ is k-dimensional white noise

### Spectral Granger Causality

Granger causality can be decomposed into frequency bands:

$$f_{X \rightarrow Y}(\omega) = \ln\left(\frac{S_Y(\omega)}{\tilde{S}_Y(\omega)}\right)$$

where:
- $S_Y(\omega)$ is the power spectrum of Y
- $\tilde{S}_Y(\omega)$ is the intrinsic spectrum of Y after removing X's influence
- $\omega$ is the frequency

This allows examination of causality in specific frequency bands (e.g., delta, theta, alpha, beta, gamma).

## Parameters and Their Effects

### 1. Model Order (Lag Length, p)

**Definition**: The number of past time points used for prediction.

**Selection Methods**:
- Akaike Information Criterion (AIC): $AIC = 2k - 2\ln(L)$
- Bayesian Information Criterion (BIC): $BIC = k\ln(n) - 2\ln(L)$
- Hannan-Quinn Criterion (HQC): $HQC = 2k\ln(\ln(n)) - 2\ln(L)$

where k = number of parameters, n = sample size, L = likelihood.

**Effects**:
- **Too Small**: Underfit the model, miss true causal relationships, increased false negatives
- **Too Large**: Overfit the model, detect spurious causality, increased false positives, reduced statistical power
- **Optimal**: Balances model complexity and predictive accuracy

**Practical Guidelines**:
- EEG data (200 Hz): typically 1-100 ms (2-20 samples)
- Start with multiple criteria (AIC, BIC) and validate
- Consider physiological constraints (synaptic delays ~5-50 ms)

### 2. Sampling Frequency

**Definition**: Number of samples per second (Hz).

**Effects**:
- **Higher Frequency**: 
  - Better temporal resolution
  - Captures faster dynamics
  - Requires more data for stable estimates
  - Computationally expensive
- **Lower Frequency**: 
  - Reduced temporal resolution
  - May miss fast interactions
  - More stable estimates with less data
  - Faster computation

**For 200 Hz EEG**:
- Excellent temporal resolution (5 ms per sample)
- Captures all physiologically relevant frequencies
- Suitable for detecting rapid causal interactions

### 3. Window Length (Segment Duration)

**Definition**: Duration of data segments used for analysis.

**Effects**:
- **Shorter Windows** (e.g., 1-2 seconds):
  - Capture non-stationary dynamics
  - Higher temporal resolution of connectivity changes
  - Less stable estimates (higher variance)
  - Risk of overfitting
  
- **Longer Windows** (e.g., 10-60 seconds):
  - More stable estimates
  - Assume stationarity within window
  - Lower temporal resolution
  - Better for steady-state connectivity

**Recommendations**:
- Resting state: 10-30 seconds
- Balance between stationarity assumption and statistical power
- Use overlapping windows for smoother transitions

### 4. Frequency Band Selection

**Definition**: Specific frequency ranges for spectral analysis.

**Standard EEG Bands**:
- Delta (0.5-4 Hz): Deep sleep, attention
- Theta (4-8 Hz): Memory, emotion
- Alpha (8-13 Hz): Relaxed wakefulness
- Beta (13-30 Hz): Active thinking, focus
- Gamma (30-100 Hz): Cognitive processing

**Effects**:
- **Band-Specific Analysis**: Reveals frequency-dependent connectivity
- **Broadband Analysis**: Overall connectivity patterns
- Different cognitive processes operate in different bands
- Infant EEG shows different frequency profiles than adults

### 5. Preprocessing Parameters

#### Filtering
**High-Pass Filter**:
- Removes slow drifts and DC offset
- Typical: 0.5-1 Hz
- **Effect**: Too high removes slow oscillations of interest

**Low-Pass Filter**:
- Removes high-frequency noise
- Typical: 30-50 Hz for EEG
- **Effect**: Too low removes relevant neural signals

**Notch Filter**:
- Removes power line noise (50/60 Hz)
- **Effect**: Essential for clean data, minimal impact on connectivity

#### Detrending
- Removes linear or polynomial trends
- **Effect**: Improves stationarity, reduces spurious causality

#### Normalization
- Z-score normalization: $(X - \mu) / \sigma$
- **Effect**: Ensures comparability across channels and subjects

### 6. Stationarity Requirements

**Definition**: Statistical properties remain constant over time.

**Testing**:
- Augmented Dickey-Fuller (ADF) test
- Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test

**Effects**:
- **Non-Stationary Data**: Can produce spurious causality
- **Differencing**: Makes data stationary but may remove information
- **Windowing**: Assumes local stationarity

**Solutions**:
- Short sliding windows
- Adaptive methods
- Time-varying Granger causality

### 7. Multiple Comparison Correction

**Problem**: Testing many connections increases false positive rate.

**Methods**:
- **Bonferroni**: Divide alpha by number of tests (conservative)
- **False Discovery Rate (FDR)**: Controls expected proportion of false positives
- **Permutation Testing**: Data-driven null distribution

**Effects**:
- **No Correction**: High false positive rate
- **Too Conservative**: Miss true connections (false negatives)
- **Appropriate Correction**: Balances sensitivity and specificity

### 8. Surrogate Data Testing

**Purpose**: Establish statistical significance against null hypothesis.

**Methods**:
- **Phase Randomization**: Preserves power spectrum
- **Time Shifting**: Breaks temporal relationships
- **Block Permutation**: Preserves local structure

**Effects**:
- Provides robust significance thresholds
- Controls for autocorrelation structure
- More appropriate than parametric tests for non-Gaussian data

## Application to Infant Resting State EEG

### Why Granger Causality is Well-Suited

#### 1. Temporal Dynamics
**Dataset Characteristics**:
- Continuous resting state recordings (528-1784 seconds)
- High temporal resolution (200 Hz)
- Captures spontaneous brain activity

**Granger Causality Advantages**:
- Designed for time series analysis
- Exploits temporal structure
- Detects directional information flow
- High temporal resolution matches fast neural dynamics

#### 2. Multivariate Analysis
**Dataset Characteristics**:
- 19-21 EEG channels
- 10-10 electrode placement system
- Spatial coverage of scalp

**Granger Causality Advantages**:
- Naturally handles multivariate systems
- Reveals network-level connectivity
- Identifies hub regions and information pathways
- Conditional analysis controls for confounds

#### 3. Developmental Neuroscience
**Research Context**:
- Brain development in first year of life
- Rapid changes in connectivity patterns
- Critical period for network formation

**Granger Causality Insights**:
- Maps effective connectivity networks
- Tracks developmental changes in directionality
- Identifies emerging causal pathways
- Relates to structural development (synaptogenesis, myelination)

#### 4. Resting State Paradigm
**Dataset Characteristics**:
- No task structure
- Spontaneous oscillations
- Eyes closed conditions (primarily)

**Granger Causality Advantages**:
- Does not require stimulus-locked activity
- Analyzes intrinsic connectivity
- Reveals default mode dynamics
- Captures spontaneous information flow

#### 5. Frequency-Specific Analysis
**Dataset Characteristics**:
- Filtered 0.5-30 Hz
- Captures multiple frequency bands
- Different bands dominate at different ages

**Granger Causality Advantages**:
- Spectral decomposition available
- Band-specific connectivity patterns
- Reveals frequency-dependent interactions
- Links to oscillatory mechanisms

### Specific Advantages for This Dataset

#### Multiple Sessions
- **18 subjects with 2 recordings**: Track developmental changes
- **3 subjects with 3 recordings**: Detailed longitudinal analysis
- **1 subject with 4 recordings**: Case study of development
- **Granger causality**: Compare connectivity patterns across ages

#### Age-Related Analysis
- **Variable ages (first year)**: Cross-sectional and longitudinal
- **Granger causality**: Map maturation of directional connectivity
- Expected patterns: Increasing long-range connectivity, hemispheric specialization

#### Clean Annotations
- **Derivatives with visual inspection**: Quality control
- **Eyes closed/open markers**: State-dependent connectivity
- **Granger causality**: State-specific network analysis

#### BIDS Format
- **Standardized structure**: Reproducible analysis
- **Rich metadata**: Control for confounds
- **Electrode coordinates**: Spatial interpretation of results

### Expected Findings

#### 1. Developmental Trends
- Increased long-range causal connections with age
- Shift from posterior to anterior dominance
- Emergence of feed-forward and feedback pathways
- Strengthening of interhemispheric connections

#### 2. Frequency-Specific Patterns
- Delta band: Long-range synchronization
- Theta band: Memory-related pathways
- Alpha band: Posterior dominance
- Beta/Gamma: Local processing networks

#### 3. Network Architecture
- Hub identification (highly connected regions)
- Modular organization
- Small-world properties
- Hierarchy of information flow

#### 4. Hemispheric Asymmetry
- Right hemisphere predominance in infancy
- Development of left-lateralized functions
- Interhemispheric coordination patterns

### Methodological Workflow

#### Phase 1: Data Preparation
1. Load EEG data (EDF format)
2. Apply artifact rejection using annotations
3. Segment into analysis windows
4. Apply preprocessing (filtering, detrending)
5. Check stationarity

#### Phase 2: Model Selection
1. Determine optimal model order (AIC, BIC)
2. Validate across subjects
3. Consider physiological constraints
4. Test sensitivity to parameter choices

#### Phase 3: Granger Causality Estimation
1. Fit VAR models
2. Compute pairwise/conditional GC
3. Generate spectral GC for frequency bands
4. Apply surrogate testing for significance

#### Phase 4: Statistical Analysis
1. Multiple comparison correction (FDR)
2. Group-level statistics
3. Age correlation analysis
4. Session comparison (longitudinal subjects)

#### Phase 5: Visualization and Interpretation
1. Connectivity matrices
2. Network graphs
3. Topographic maps
4. Frequency-resolved plots
5. Developmental trajectories

## Practical Considerations

### Computational Requirements

**Memory**:
- VAR models: $O(k^2 \times p)$ where k = channels, p = lags
- 21 channels, lag 20: Manageable on standard hardware

**Time Complexity**:
- Per subject: Minutes to hours depending on segment length
- 103 subjects: Batch processing recommended
- Parallelization: Across subjects or frequency bands

### Software Implementations

**Python**:
- `statsmodels`: VAR models and Granger tests
- `mvgc`: Multivariate Granger causality toolbox
- `nitime`: Spectral Granger causality
- `MNE-Python`: EEG preprocessing and connectivity

**MATLAB**:
- MVGC Toolbox: Comprehensive multivariate analysis
- SIFT: Source Information Flow Toolbox
- FieldTrip: EEG/MEG connectivity analysis
- BSMART: Spectral analysis

**R**:
- `vars`: VAR modeling
- `lmtest`: Granger causality tests
- `grangers`: Spectral methods

### Quality Control

1. **Visual Inspection**: Check residuals for whiteness
2. **Model Diagnostics**: Stability, residual autocorrelation
3. **Consistency Check**: Results stable across parameter variations
4. **Biological Plausibility**: Compare with known developmental patterns
5. **Replication**: Validate on held-out subjects

### Common Pitfalls

1. **Overfitting**: Using too many lags
2. **Non-Stationarity**: Violates model assumptions
3. **Common Input**: Can create spurious causality
4. **Volume Conduction**: EEG signals mix at scalp
5. **Multiple Comparisons**: Inflated false positive rate

### Solutions

1. **Regularization**: Ridge regression, LASSO for high-dimensional data
2. **Short Windows**: Local stationarity assumption
3. **Conditional GC**: Control for common influences
4. **Source Space**: Estimate GC at source level (requires head model)
5. **Proper Statistics**: FDR correction, permutation testing

## Interpretation Guidelines

### What Granger Causality Tells Us

**Does Tell**:
- Direction of predictive information flow
- Strength of statistical association
- Frequency-specific interactions
- Network structure and hubs

**Does Not Tell**:
- Direct anatomical connections
- True causal mechanisms
- Instantaneous interactions
- Subcortical involvement (with scalp EEG)

### Validation Approaches

1. **Cross-Validation**: Test-set performance
2. **Surrogate Analysis**: Null hypothesis testing
3. **Comparison with Anatomy**: Known white matter tracts
4. **Literature Agreement**: Published developmental patterns
5. **Convergent Methods**: Compare with other connectivity measures

## References

### Foundational Papers

1. Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3), 424-438.

2. Geweke, J. (1982). Measurement of linear dependence and feedback between multiple time series. Journal of the American Statistical Association, 77(378), 304-313.

3. Bressler, S. L., & Seth, A. K. (2011). Wiener-Granger causality: A well established methodology. NeuroImage, 58(2), 323-329.

### EEG Applications

4. Kamiński, M., Ding, M., Truccolo, W. A., & Bressler, S. L. (2001). Evaluating causal relations in neural systems: Granger causality, directed transfer function and statistical assessment of significance. Biological Cybernetics, 85(2), 145-157.

5. Ding, M., Chen, Y., & Bressler, S. L. (2006). Granger causality: Basic theory and application to neuroscience. Handbook of Time Series Analysis, 437-460.

6. Seth, A. K., Barrett, A. B., & Barnett, L. (2015). Granger causality analysis in neuroscience and neuroimaging. Journal of Neuroscience, 35(8), 3293-3297.

### Developmental Studies

7. Bosch-Bayard, J., et al. (2022). EEG effective connectivity during the first year of life mirrors brain synaptogenesis, myelination, and early right hemisphere predominance. NeuroImage, 252, 119035.

8. Gao, W., et al. (2015). Functional connectivity of the infant human brain: Plastic and modifiable. The Neuroscientist, 21(2), 169-184.

### Methodological Advances

9. Barnett, L., & Seth, A. K. (2014). The MVGC multivariate Granger causality toolbox: A new approach to Granger-causal inference. Journal of Neuroscience Methods, 223, 50-68.

10. Dhamala, M., Rangarajan, G., & Ding, M. (2008). Analyzing information flow in brain networks with nonparametric Granger causality. NeuroImage, 41(2), 354-362.

### Statistical Considerations

11. Ding, M., Bressler, S. L., Yang, W., & Liang, H. (2000). Short-window spectral analysis of cortical event-related potentials by adaptive multivariate autoregressive modeling: Data preprocessing, model validation, and variability assessment. Biological Cybernetics, 83(1), 35-45.

12. Cui, J., Xu, L., Bressler, S. L., Ding, M., & Liang, H. (2008). BSMART: A Matlab/C toolbox for analysis of multichannel neural time series. Neural Networks, 21(8), 1094-1104.

---