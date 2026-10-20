# Class Size and Student Achievement: Regression Discontinuity with Maimonides' Rule

A Python replication and extension of **Angrist and Lavy (1999)**. Israeli public schools split a grade into an additional class whenever enrollment crosses a multiple of 40, so class size drops discontinuously at enrollments of 41, 81, 121, and so on. These rule-induced jumps identify the causal effect of class size on test scores.

## Design

| | |
|---|---|
| **Running variable** | Grade enrollment |
| **Cutoffs** | 41, 81, 121, … (Maimonides' rule) |
| **Treatment** | Average class size |
| **Outcomes** | Average math and verbal scores |
| **Control** | Share of disadvantaged students |

## Main result

Fuzzy RD at the first cutoff (enrollment 20–60, local linear trend, enrollment above 41 as instrument for class size):

| | Effect of one extra student per class on average math score |
|---|---|
| Fuzzy RD / 2SLS | **−0.38 (SE 0.15)** |

Naive OLS gives a *positive* association between class size and scores, because larger classes are found in larger, more advantaged schools. The discontinuity design reverses the sign, consistent with Angrist and Lavy's conclusion that smaller classes raise achievement.

## Methods

- Replication of the descriptive statistics (Table I) and OLS results (Table II) of the original paper
- Graphical RD analysis: class size and scores against enrollment, with local averages around each threshold
- Sharp RD at the 41 cutoff with linear trends and covariates
- Fuzzy RD / 2SLS using the rule-predicted class size
- Validity check: no jump in the share of disadvantaged students at the cutoff

## Repository structure

```
analysis.md   # full analysis: code, regression output and interpretation
```

Figures and tables are generated when the code is run; the figure files themselves are not stored in the repository.

## Reproducing

The `maimonides.dta` file comes from the Angrist and Lavy (1999) replication data ([Angrist Data Archive](https://economics.mit.edu/people/faculty/josh-angrist/angrist-data-archive)).

```bash
pip install pandas numpy statsmodels matplotlib seaborn
```

## Reference

Angrist, J. D., and Lavy, V. (1999). Using Maimonides' Rule to Estimate the Effect of Class Size on Scholastic Achievement. *Quarterly Journal of Economics*, 114(2), 533–575.

## Context

Developed during the MSc in Economics at Sharif University of Technology (Applied Econometrics, Spring 2025).

---
**Shayan Pourmasoomi** · MSc Economics, Sharif University of Technology · [LinkedIn](https://www.linkedin.com/in/shayan-pourmasoomi-8b6206389)
