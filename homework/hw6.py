import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats import proportion as sm_prop
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import roc_curve, roc_auc_score

#1
df = pd.read_csv("bodytemp.csv")
df["temp_level"] = (df["temp"] >= 98.6).astype(int)

print(df["temp_level"].value_counts())
print(df.head(10))

#2
print("\n")
print(pd.crosstab(df["sex"], df["temp_level"]))
print(pd.crosstab(df["sex"], df["temp_level"], normalize="index"))

#3
s1, n1 = 14, 65   # males 
s2, n2 = 35, 65   #female

count = np.array([s1, s2])
nobs = np.array([n1, n2])

z_stat_2, p_value_2 = sm_prop.proportions_ztest(count=count, nobs=nobs, alternative="two-sided")

ci1 = sm_prop.proportion_confint(s1, n1, alpha=0.05, method="normal")
ci2 = sm_prop.proportion_confint(s2, n2, alpha=0.05, method="normal")

p1 = s1 / n1
p2 = s2 / n2
risk_difference = p2 - p1   # female - male

print("\n")
print("Two-sample proportion test:")
print("p_male:", p1)
print("p_female:", p2)
print("risk difference:", risk_difference)
print("z-statistic:", z_stat_2)
print("p-value:", p_value_2)
print("95% CI males:", ci1)
print("95% CI females:", ci2)

#4
df["female"] = (df["sex"] == 2).astype(int)

m = smf.glm("temp_level ~ female", data=df, family=sm.families.Binomial()).fit()
print("\n")
print(m.summary())

params = m.params
conf = m.conf_int()
or_df = pd.DataFrame({
    "OR": np.exp(params),
    "CI_lower": np.exp(conf[0]),
    "CI_upper": np.exp(conf[1])
})
print(or_df)

df["prob"] = m.predict(df)
print("c-statistic:", roc_auc_score(df["temp_level"], df["prob"]))

#5
m2 = smf.glm("temp_level ~ female + heart_rate", data=df, family=sm.families.Binomial()).fit()
print("\n")
print(m2.summary())

params = m2.params
conf = m2.conf_int()
or_df = pd.DataFrame({
    "OR": np.exp(params),
    "CI_lower": np.exp(conf[0]),
    "CI_upper": np.exp(conf[1])
})
print(or_df)

beta_hr = m2.params["heart_rate"]
se_hr = m2.bse["heart_rate"]
or_10 = np.exp(beta_hr * 10)

z_crit = stats.norm.ppf(0.975)
or_10_ci = (np.exp((beta_hr - z_crit * se_hr) * 10),
            np.exp((beta_hr + z_crit * se_hr) * 10))

print("OR for heart rate (10-beat increase):", or_10)
print("95% CI:", or_10_ci)

df["prob2"] = m2.predict(df)
print("c-statistic:", roc_auc_score(df["temp_level"], df["prob2"]))

#6
lr_stat = m2.deviance - m.deviance   # note: m is the smaller model
lr_stat = m.deviance - m2.deviance
p_lr = stats.chi2.sf(lr_stat, df=1)
print("\n")
print("LR statistic:", lr_stat, " p =", p_lr)

fpr, tpr, thresholds = roc_curve(df["temp_level"], df["prob2"])
auc_value = roc_auc_score(df["temp_level"], df["prob2"])

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_value:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="No discrimination")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model 2")
plt.legend(loc="lower right")
plt.savefig("roc_model2.png")
plt.show()