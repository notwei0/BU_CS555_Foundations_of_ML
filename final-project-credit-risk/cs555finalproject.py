import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.formula.api as smf

df = pd.read_csv("cleaned_german_credit_data.csv")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for ax, var, title in [
    (axes[0, 0], "status",         "bad-risk rate by checking account status"),
    (axes[0, 1], "property",       "bad-risk rate by property"),
    (axes[1, 0], "credit_history", "bad-risk rate by credit history"),
]:
    rate = df.groupby(var)["bad_risk"].mean()
    ax.bar(rate.index.astype(str), rate.values)
    ax.axhline(df["bad_risk"].mean(), linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel(var)
    ax.set_ylabel("precentage of samples that are bad")
    ax.set_ylim(0, 0.7)

ax = axes[1, 1]
ax.boxplot([df.loc[df["bad_risk"] == 0, "age"],
            df.loc[df["bad_risk"] == 1, "age"]])
ax.set_xticks([1, 2])
ax.set_xticklabels(["good (0)", "bad (1)"])
ax.set_title("Age by credit outcome")
ax.set_ylabel("age (years)")

plt.tight_layout()
plt.savefig("descriptives.png", dpi=200)
plt.show()

# fit a multiple logistic regression model using statsmodels 
model = smf.glm("bad_risk ~ C(status) + C(property) + C(credit_history) + age", data=df, family=sm.families.Binomial()).fit()
print(model.summary())

#Wald test for significance of each predictor
print("\nwald test for significance of each predictor:")
print(model.wald_test_terms(scalar=True))

print("\nglobal Wald test (all predictors):")
R = np.eye(len(model.params))[1:]
print(model.wald_test(R, scalar=True)) #tweaked based off module 6"s code

# for loop that creates a fitted model equation instead of having to go through all coefficients and manually write out the equation, which is helpful for larger models with many predictors

terms = []
for name, coef in model.params.items():
    if name == "Intercept":
        terms.append(f"{coef:.3f}")
    else:
        clean = name.replace("C(", "I(").replace(")[T.", "=").replace("]", ")")
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.3f}*{clean}")

print("\nlog-odds(bad risk) =", " ".join(terms))

#odds ratios and 95% confidence intervals
#used module 6's code from Blackboard as reference
params = model.params
conf = model.conf_int()  # default: normal-approx CI

or_df = pd.DataFrame({
    "OR": np.exp(params),
    "CI_lower": np.exp(conf[0]),
    "CI_upper": np.exp(conf[1]),
    "p-values": model.pvalues,
})
print("\nodds Ratios with 95% CI:")
print(or_df)

#odds ratio for a 10 year increase in age since age is a continuous not a categorical variable
#used module 6's code from Blackboard as reference
print(f"\nOR: {np.exp(model.params['age'] * 10)}")
print(f"95% CI: ({np.exp(model.conf_int().loc['age', 0] * 10):.3f}, {np.exp(model.conf_int().loc['age', 1] * 10):.3f})")

#ROC and AUC
#used module 6's code from Blackboard as reference
df["prob"] = model.predict(df)

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(df["bad_risk"], df["prob"])
auc_value = roc_auc_score(df["bad_risk"], df["prob"])

print("\nROC / AUC for multiple logistic regression model:")
print("AUC (c-statistic):", round(auc_value, 3))

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_value:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="No discrimination")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()

# predicted risk for two applicant profiles
# used module 6's code from Blackboard as reference
b = model.params

# favorable: >= 200 DM / salary, all credits at this bank paid, no known property, age 40
logit_fav = (b["Intercept"] + b["C(status)[T.4]"]
             + b["C(credit_history)[T.4]"] + b["age"] * 40)
risk_fav = np.exp(logit_fav) / (1 + np.exp(logit_fav))

# Unfavorable: no checking account, past delay, real estate, age 25
logit_unfav = (b["Intercept"] + b["C(property)[T.4]"] + b["age"] * 25)
risk_unfav = np.exp(logit_unfav) / (1 + np.exp(logit_unfav))

print("\n")
print("favorable profile:  ", round(risk_fav, 4))
print("unfavorable profile:", round(risk_unfav, 4))
print("Risk ratio:", round(risk_unfav / risk_fav, 1))