# Regression Discontinuity Design (RDD) — Class Size and Student Achievement
### Evidence from the Maimonides Rule (Angrist & Lavy, 1999)

This repository contains a **Python implementation of Regression Discontinuity Design (RDD)** methods to study the causal effect of **class size on student math achievement**, using Israeli school data from **Angrist and Lavy (1999)**.

The project is based on **Problem Set 08 – Applied Metrics (MSc), Spring 2025**, and covers **OLS, Sharp RDD, Fuzzy RDD, diagnostics, and robustness checks**.

---

## Project Overview

The Israeli education system follows the **Maimonides rule**, which caps class size at **40 students**. When enrollment exceeds multiples of 40, an additional class is created, generating **discontinuities in class size**.

We exploit these discontinuities using **RDD** to estimate the causal effect of class size on **average math test scores**.

- **Running variable**: School enrollment  
- **Cutoff(s)**: 41, 81, 121, …  
- **Treatment**: Larger (smaller) class size  
- **Outcome**: Average math score (`avgmath`)

---

## Data

Input dataset:
- `maimonides.dta`

Key variables:
- `classize` — Average class size  
- `avgmath` — Average math score  
- `avgverb` — Average verbal score  
- `enrollment` — School enrollment (running variable)  
- `perc_disadvantaged` — Share of disadvantaged students  

---

## Methods Implemented

### 1. Descriptive Statistics
- Summary statistics for key variables
- Replication of Table I in Angrist & Lavy (1999)

### 2. Graphical Analysis
- Scatter plot of class size vs. enrollment
- Local averages (binscatter)
- Visualization of enrollment thresholds

### 3. OLS Regressions
- Conditional correlation between class size and math scores
- Replication of columns (4)–(6) of Table II

### 4. Sharp Regression Discontinuity Design
- Single cutoff at enrollment = 41
- Sample restricted to enrollment ∈ [20, 60]
- Linear trends and covariate controls
- Extension to all thresholds using **predicted class size**

### 5. Fuzzy Regression Discontinuity Design
- First-stage: cutoff as an instrument for class size
- Second-stage: IV estimation of causal effect
- Implementation using:
  - Single cutoff
  - All thresholds (replicating Table IV)

### 6. Sensitivity and Robustness
- Inclusion/exclusion of covariates
- Bandwidth sensitivity (±10, ±15, ±20)
- Linear vs quadratic polynomials

### 7. RDD Diagnostics (Lee & Lemieux, 2010)
- Histogram of enrollment (manipulation test)
- RD binscatter plot
- Polynomial trend comparison
- Placebo RDD using baseline covariate (`perc_disadvantaged`)

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy
