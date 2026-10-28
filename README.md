# CS555 Foundations of Machine Learning — Boston University

Coursework from CS555 (Boston University Metropolitan College), by Wei Zhao: applied statistics and regression modeling in Python.

## Final project: Credit Risk Modeling with Logistic Regression

A multiple logistic regression model that predicts credit default from four applicant characteristics (checking account status, property, credit history, age) using 1,000 historical loan records from the corrected *South German Credit* dataset (UCI Machine Learning Repository).

- All four predictors were statistically significant (global Wald test, p < 0.001), with a c-statistic (ROC AUC) of 0.76.
- The widely used older version of this dataset contains documented coding errors, so the project uses the corrected release and checks every variable encoding against the official code table.
- The positive association between collateral and default is treated as likely selection bias (the sample only contains approved loans), and because defaults were deliberately oversampled, results are reported as odds ratios rather than absolute probabilities.

| File | What it does |
|---|---|
| `dataclean.py` | Downloads the corrected dataset from UCI, renames the German columns to English, saves `cleaned_german_credit_data.csv` |
| `cs555finalproject.py` | Descriptive plots, the logistic regression model (statsmodels GLM), and the ROC curve |
| `trial_error.py` | Exploratory comparison with a linear (OLS) model |

See [`final-project-credit-risk/`](final-project-credit-risk/).

## Homework

All homework scripts and their datasets are in [`homework/`](homework/).

| File | Topics |
|---|---|
| `hw1.py` | Descriptive statistics and histograms |
| `hw2.py` | Confidence intervals and t-tests |
| `hw3.py` | Correlation and simple linear regression |
| `hw4.py` | Linear regression and residual analysis |
| `hw5.py` | One-way ANOVA and Tukey comparisons |
| `hw6.py` | Logistic regression and ROC curves |

**Tools:** Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, matplotlib

**Run:** `pip install pandas numpy scipy statsmodels scikit-learn matplotlib`, then run a script from inside its folder (the scripts read their CSVs by relative path).
