import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_1samp, ttest_ind

#1  creates the boxplot and gets the data
df = pd.read_csv("calories.csv")
p = df[df["group"] == "participant"]["calories"]
nonp = df[df["group"] == "non-participant"]["calories"]
#creates boxplot for calories by group
df.boxplot(column="calories", by="group")
plt.xlabel("group")
plt.ylabel("calories")
plt.title("Calories by group")
plt.suptitle("")
plt.savefig("boxplot.png")
plt.show()

#prints the data
print(df.groupby("group")["calories"].describe())

#2 + 3 performs a t-test and calculates the confidence interval
# vs 425
print("\n")
print("2 + 3")
x = p.values
mu0 = 425  #sets null hypothesis mean

t_stat, p_val_two_sided = ttest_1samp(x, popmean=mu0)
print(f"t = {t_stat:.3f}, two-sided p = {p_val_two_sided:.3f}")

alternative = "greater"
if alternative == "less":
    p_val = p_val_two_sided / 2 if t_stat < 0 else 1 - p_val_two_sided / 2
elif alternative == "greater":
    p_val = p_val_two_sided / 2 if t_stat > 0 else 1 - p_val_two_sided / 2
else:
    p_val = p_val_two_sided
print(f"one-sided p = {p_val:.3f}")

# critical values for the decision rules (alpha = 0.05)
n = len(x)
dfree = n - 1  #degrees of freedom, n - 1
print(f"#2 critical value (two-sided): {t.ppf(1 - 0.05/2, dfree):.3f}")
print(f"#3 critical value (one-sided): {t.ppf(1 - 0.05, dfree):.3f}")

# 90% confidence interval
conf_level = 0.90
alpha = 1 - conf_level
xbar = np.mean(x)
s = np.std(x, ddof=1)
se = s / np.sqrt(n)
t_crit = t.ppf(1 - alpha/2, dfree)
lower = xbar - t_crit * se
upper = xbar + t_crit * se
print(f"90% CI: ({lower:.1f}, {upper:.1f})")

#4
print("\n")
print("4")
t_stat2, p_two2 = ttest_ind(p.values, nonp.values, equal_var=False)
p_val2 = p_two2/2 if t_stat2 > 0 else 1 - p_two2/2
print(f"t = {t_stat2:.3f}, one-sided p = {p_val2:.3f}")

x1, x2 = p.values, nonp.values
n1, n2 = len(x1), len(x2)
m1, m2 = np.mean(x1), np.mean(x2)
s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)

se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
dfree2 = min(n1, n2) - 1  #smaller sample size - 1, per module examples
print(f"#4 critical value (one-sided, df=21): {t.ppf(1 - 0.05, dfree2):.3f}")

t_crit2 = t.ppf(1 - 0.10/2, dfree2)  #90% CI multiplier
lower2 = (m1 - m2) - t_crit2 * se_diff
upper2 = (m1 - m2) + t_crit2 * se_diff
print(f"90% CI for difference: ({lower2:.1f}, {upper2:.1f})")