Here is a simplified, step-by-step roadmap in plain English for applying Granger Causality to this dataset.

[cite_start]This guide follows the principles of the research paper[cite: 6]. [cite_start]The single most important rule, stressed by the authors, is that you **cannot** apply connectivity analysis directly to the raw scalp EEG data[cite: 120]. [cite_start]Doing so will produce "spurious connections" due to the volume conduction effect[cite: 120].

This roadmap describes how to replicate the paper's preprocessing pipeline to generate clean "source-level" signals, which you can then analyze with Granger Causality.

---

## Roadmap: Applying Granger Causality to Infant EEG Source Data

### 1. Data Preparation and Epoching

Your first step is to extract the exact, pre-cleaned data segments identified by the original researchers.

* Navigate to the `derivatives/NeuronicEEG/` folder.
* For a given subject, open their `..._annotations.tsv` file. This file contains the `onset` and `duration` for every artifact-free, 2.56-second segment the researchers approved for analysis.
* Use these onset times to find and slice the corresponding raw `.edf` file (located in `sub-NORB#####/ses-#/eeg/`).
* [cite_start]This process will give you a collection of small "epochs," each 2.56 seconds long[cite: 112].
* As you noted, the data is already bandpass filtered (0.5-30 Hz), so no further filtering is needed.

---

### 2. Source Localization (Moving to the Brain)

Now you must project the 19-channel scalp data into the 3D brain "source space." This requires a **Forward Model** and an **Inverse Model**.

* **Forward Model:** This is a physics-based model of how electrical signals travel from the brain's gray matter to the scalp. [cite_start]You will need to construct this model using an infant head template (the paper used a custom neonate template [cite: 141]) and the electrode coordinates from the `..._electrodes.tsv` file.
* [cite_start]**Inverse Model (sLoreta):** Once you have the forward model, you create an "inverse operator" using the **sLoreta** method, as specified in the paper[cite: 139]. This operator is the mathematical tool that estimates the brain activity that *caused* the scalp signals.
* **Apply the Model:** Apply this sLoreta operator to every 2.56-second epoch. [cite_start]The output will no longer be 19 channel signals, but thousands of "virtual electrodes" (sources) across the brain's gray matter[cite: 145].

---

### 3. Signal Unmixing (The Critical Step)

[cite_start]The paper warns that even sLoreta's output suffers from a "mixing (leakage) problem"[cite: 122]. You must now "unmix" these source-level signals.

* [cite_start]The authors used a specific **unmixing algorithm** (from Biscay et al., 2018) [cite: 124] to solve this.
* [cite_start]This algorithm mathematically "sharpens" the signals at the source level, ensuring the activity from one brain region is not leaking into and contaminating its neighbors[cite: 137].
* You must apply this unmixing algorithm (or a similar leakage-correction method) to your sLoreta source data.

---

### 4. Region of Interest (ROI) Time Series Extraction

You now have clean, unmixed source data. The next step is to group the thousands of sources into the 16 Regions of Interest (ROIs) used in the study.

* [cite_start]Go to **Table 2** in the research paper[cite: 271]. This table provides the exact MNI coordinates for the center of all 16 ROIs (e.g., 'Left Thalamus' at [-14, -14, 13]).
* [cite_start]For each of the 16 ROIs, select all the unmixed source signals that fall within a **1 cm diameter sphere** around that coordinate[cite: 192].
* Average the signals within each sphere.
* The result is your final, clean data: **16 distinct time series** (one for each ROI) for each 2.56-second epoch.

---

### 5. Applying Granger Causality

You are finally ready to perform the connectivity analysis.

* Take your 16 clean ROI time series for a single epoch.
* Fit a **multivariate autoregressive (MVAR) model** to this data. This model predicts the future of each time series based on the past of *all* other time series.
* You must first select a "model order" (the number of past time points, or 'lags') to use for the prediction. This is often done using a statistical criterion like BIC or AIC.
* Once the MVAR model is fit, you can perform the **Granger Causality test**. This is a statistical test (an F-test) that determines if the past values of one ROI (e.g., 'Right Thalamus') significantly help predict the future values of another ROI (e.g., 'Left Precentral').
* A significant p-value (e.g., p < 0.05) suggests a directed, causal connection.

---

### 6. Population-Level Analysis (The Paper's Goal)

To get the *findings* from the paper, you must aggregate your results.

* Repeat steps 1-5 for *every* 2.56s epoch for *every* subject in the dataset.
* For each subject, find their age by opening the `..._scans.tsv` file and reading the `age_acq_time` column.
* [cite_start]You can now replicate the paper's final analysis: run a **robust linear regression** [cite: 200] where the independent variable is the infant's age and the dependent variable is the Granger Causality F-statistic for a specific connection.
* [cite_start]This will show you which connections get stronger or weaker as the infants get older [cite: 408-410]. [cite_start]Remember to correct your results for multiple comparisons using the **False Discovery Rate (FDR)**, just as the authors did[cite: 199, 203].